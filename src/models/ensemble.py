"""
Ensemble methods with proper out-of-fold predictions to prevent overfitting.

Key Principle:
- Base models should NEVER see the same data they're making predictions on
- Use k-fold cross-validation to generate out-of-fold (OOF) predictions
- Meta-learner trains on OOF predictions (held-out data)
- This prevents the meta-learner from learning on overfitted base predictions
"""

from typing import List, Tuple, Callable, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from loguru import logger
import xgboost as xgb
import lightgbm as lgb


class StackingEnsemble:
    """
    Stacking ensemble with proper out-of-fold predictions.

    This prevents data leakage by ensuring base model predictions
    used for meta-learning are from held-out folds.

    Example:
        ensemble = StackingEnsemble(
            base_models=[xgb_trainer, lgb_trainer],
            meta_model=LogisticRegression()
        )
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict_proba(X_test)
    """

    def __init__(
        self,
        base_models: List[Dict],
        meta_model=None,
        n_splits: int = 5,
        random_state: int = 42
    ):
        """
        Initialize stacking ensemble.

        Args:
            base_models: List of dicts with 'name', 'trainer' function
            meta_model: Meta-learner (default: LogisticRegression)
            n_splits: Number of CV folds for OOF predictions
            random_state: Random seed for reproducibility
        """
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(
            random_state=random_state,
            max_iter=1000
        )
        self.n_splits = n_splits
        self.random_state = random_state

        # Trained models storage
        self.base_models_fitted = {model['name']: [] for model in base_models}
        self.meta_model_fitted = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        verbose: bool = True
    ) -> 'StackingEnsemble':
        """
        Fit stacking ensemble with out-of-fold predictions.

        Process:
        1. Split training data into k folds (time-series aware)
        2. For each fold:
           - Train base models on train portion
           - Predict on validation portion (OOF predictions)
        3. Combine all OOF predictions
        4. Train meta-learner on OOF predictions
        5. Retrain base models on full training set for final predictions

        Args:
            X_train: Training features
            y_train: Training target
            verbose: Print progress logs

        Returns:
            self (fitted ensemble)
        """
        if verbose:
            logger.info("="*80)
            logger.info("STACKING ENSEMBLE WITH OUT-OF-FOLD PREDICTIONS")
            logger.info("="*80)
            logger.info(f"\nTraining data: {X_train.shape}")
            logger.info(f"Base models: {[m['name'] for m in self.base_models]}")
            logger.info(f"Meta-learner: {self.meta_model.__class__.__name__}")
            logger.info(f"CV splits: {self.n_splits}")

        # Initialize OOF prediction arrays
        n_samples = len(X_train)
        n_models = len(self.base_models)
        oof_predictions = np.zeros((n_samples, n_models))

        # Time-series cross-validation (no shuffle to preserve temporal order)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        if verbose:
            logger.info(f"\n{'='*80}")
            logger.info("STEP 1: GENERATING OUT-OF-FOLD PREDICTIONS")
            logger.info("="*80)

        # Generate OOF predictions for each model
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            if verbose:
                logger.info(f"\nFold {fold_idx + 1}/{self.n_splits}")
                logger.info(f"  Train: {len(train_idx)} samples")
                logger.info(f"  Val:   {len(val_idx)} samples")

            # Split fold data
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]

            # Train each base model on this fold
            for model_idx, model_config in enumerate(self.base_models):
                model_name = model_config['name']
                trainer_func = model_config['trainer']

                if verbose:
                    logger.info(f"    Training {model_name}...")

                # Train model on fold training data
                fitted_model = trainer_func(X_fold_train, y_fold_train)

                # Store fitted model for this fold
                self.base_models_fitted[model_name].append(fitted_model)

                # Generate OOF predictions on fold validation data
                if model_name.lower().startswith('xgb'):
                    # XGBoost uses DMatrix
                    dval = xgb.DMatrix(X_fold_val)
                    fold_predictions = fitted_model.predict(dval)
                elif model_name.lower().startswith('lgb') or model_name.lower().startswith('lightgbm'):
                    # LightGBM direct prediction
                    fold_predictions = fitted_model.predict(
                        X_fold_val,
                        num_iteration=fitted_model.best_iteration
                    )
                else:
                    # Scikit-learn style
                    fold_predictions = fitted_model.predict_proba(X_fold_val)[:, 1]

                # Store OOF predictions
                oof_predictions[val_idx, model_idx] = fold_predictions

        if verbose:
            logger.info(f"\n{'='*80}")
            logger.info("STEP 2: TRAINING META-LEARNER ON OOF PREDICTIONS")
            logger.info("="*80)
            logger.info(f"\nOOF predictions shape: {oof_predictions.shape}")
            logger.info("Note: These are held-out predictions (no overfitting)")

        # Train meta-learner on OOF predictions
        self.meta_model_fitted = self.meta_model.fit(oof_predictions, y_train)

        if verbose:
            logger.info(f"✓ Meta-learner trained: {self.meta_model.__class__.__name__}")
            logger.info(f"\n{'='*80}")
            logger.info("STEP 3: RETRAINING BASE MODELS ON FULL TRAINING SET")
            logger.info("="*80)

        # Retrain base models on full training set for final predictions
        self.base_models_final = {}
        for model_config in self.base_models:
            model_name = model_config['name']
            trainer_func = model_config['trainer']

            if verbose:
                logger.info(f"  Training {model_name} on full training set...")

            self.base_models_final[model_name] = trainer_func(X_train, y_train)

        if verbose:
            logger.info(f"\n{'='*80}")
            logger.success("✓ STACKING ENSEMBLE TRAINING COMPLETE")
            logger.info("="*80)
            logger.info("\nKey Points:")
            logger.info("  1. ✓ OOF predictions generated (no data leakage)")
            logger.info("  2. ✓ Meta-learner trained on held-out data")
            logger.info("  3. ✓ Base models retrained on full data for final use")
            logger.info("  4. ✓ Ready for test set predictions")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate stacked predictions for new data.

        Process:
        1. Get predictions from each base model (trained on full train set)
        2. Stack base predictions
        3. Feed to meta-learner for final predictions

        Args:
            X: Features to predict on

        Returns:
            Array of probabilities (n_samples,)
        """
        if not self.base_models_final:
            raise ValueError("Ensemble not fitted. Call .fit() first.")

        # Get base model predictions
        base_predictions = []

        for model_config in self.base_models:
            model_name = model_config['name']
            model = self.base_models_final[model_name]

            # Generate predictions based on model type
            if model_name.lower().startswith('xgb'):
                dx = xgb.DMatrix(X)
                preds = model.predict(dx)
            elif model_name.lower().startswith('lgb') or model_name.lower().startswith('lightgbm'):
                preds = model.predict(X, num_iteration=model.best_iteration)
            else:
                preds = model.predict_proba(X)[:, 1]

            base_predictions.append(preds)

        # Stack predictions
        X_meta = np.column_stack(base_predictions)

        # Meta-learner final predictions
        final_predictions = self.meta_model_fitted.predict_proba(X_meta)[:, 1]

        return final_predictions

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Generate binary predictions.

        Args:
            X: Features to predict on
            threshold: Classification threshold

        Returns:
            Binary predictions (n_samples,)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def get_oof_scores(self, y_train: pd.Series) -> Dict[str, float]:
        """
        Calculate OOF performance metrics.

        Args:
            y_train: True training labels

        Returns:
            Dictionary of metric_name -> score
        """
        # This would require storing OOF predictions during training
        # Not implemented in this version
        raise NotImplementedError("OOF scoring not yet implemented")


def create_xgb_trainer(params: Dict = None) -> Callable:
    """
    Create XGBoost trainer function.

    Args:
        params: XGBoost parameters

    Returns:
        Trainer function that takes (X, y) and returns fitted model
    """
    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist'
    }

    if params:
        default_params.update(params)

    def trainer(X_train, y_train):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(
            default_params,
            dtrain,
            num_boost_round=300,
            verbose_eval=False
        )
        return model

    return trainer


def create_lgb_trainer(params: Dict = None) -> Callable:
    """
    Create LightGBM trainer function.

    Args:
        params: LightGBM parameters

    Returns:
        Trainer function that takes (X, y) and returns fitted model
    """
    default_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'random_state': 42,
        'verbose': -1
    }

    if params:
        default_params.update(params)

    def trainer(X_train, y_train):
        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(
            default_params,
            train_data,
            num_boost_round=300,
            callbacks=[lgb.log_evaluation(period=0)]
        )
        return model

    return trainer


if __name__ == "__main__":
    # Example usage with synthetic data
    logger.info("Testing StackingEnsemble with synthetic data...")

    from sklearn.datasets import make_classification

    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )

    X_train = pd.DataFrame(X[:800], columns=[f'feat_{i}' for i in range(20)])
    y_train = pd.Series(y[:800])
    X_test = pd.DataFrame(X[800:], columns=[f'feat_{i}' for i in range(20)])
    y_test = pd.Series(y[800:])

    # Define base models
    base_models = [
        {
            'name': 'XGBoost',
            'trainer': create_xgb_trainer()
        },
        {
            'name': 'LightGBM',
            'trainer': create_lgb_trainer()
        }
    ]

    # Create and train ensemble
    ensemble = StackingEnsemble(base_models=base_models, n_splits=5)
    ensemble.fit(X_train, y_train, verbose=True)

    # Make predictions
    test_predictions = ensemble.predict_proba(X_test)

    # Calculate accuracy
    test_pred_binary = (test_predictions > 0.5).astype(int)
    accuracy = (test_pred_binary == y_test).mean()

    logger.success(f"\n✓ Test Accuracy: {accuracy:.1%}")
    logger.info("\nStackingEnsemble test complete!")
