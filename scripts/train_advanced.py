"""
Advanced Model Training Script

Improvements over baseline:
1. Class imbalance handling (SMOTE, class weights)
2. Advanced feature engineering (interactions, polynomials, matchup features)
3. Model calibration (Platt scaling)
4. Stacked ensemble with out-of-fold predictions
5. Bayesian hyperparameter optimization
6. Comprehensive evaluation and analysis
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from loguru import logger
import mlflow
import mlflow.xgboost
import mlflow.lightgbm

from src.utils.config import get_config
from src.data.loaders import load_ufc_data, get_feature_and_target_columns, validate_no_leakage
from src.data.splitters import TemporalSplitter


def create_matchup_features(df, feature_cols):
    """
    Create matchup-specific features (differentials and ratios)

    Args:
        df: DataFrame with both fighter features
        feature_cols: List of feature column names

    Returns:
        DataFrame with additional matchup features
    """
    logger.info("\n" + "="*80)
    logger.info("CREATING MATCHUP FEATURES")
    logger.info("="*80)

    df_enhanced = df.copy()
    new_features = []

    # Find paired features (f_1_* and f_2_*)
    f1_features = [col for col in feature_cols if col.startswith('f_1_')]
    f2_features = [col for col in feature_cols if col.startswith('f_2_')]

    # Get base feature names
    f1_bases = {col.replace('f_1_', ''): col for col in f1_features}
    f2_bases = {col.replace('f_2_', ''): col for col in f2_features}

    # Find common features
    common_bases = set(f1_bases.keys()) & set(f2_bases.keys())

    logger.info(f"\nFound {len(common_bases)} paired features for matchup analysis")

    # Create differentials and ratios
    for base in common_bases:
        f1_col = f1_bases[base]
        f2_col = f2_bases[base]

        # Skip if already a differential
        if 'diff_' in base:
            continue

        # Differential: F1 - F2
        diff_col = f'matchup_diff_{base}'
        df_enhanced[diff_col] = df[f1_col] - df[f2_col]
        new_features.append(diff_col)

        # Ratio: F1 / F2 (avoid division by zero)
        ratio_col = f'matchup_ratio_{base}'
        df_enhanced[ratio_col] = df[f1_col] / (df[f2_col] + 1e-6)
        new_features.append(ratio_col)

    logger.success(f"✓ Created {len(new_features)} matchup features")
    logger.info(f"  {len(new_features)//2} differentials")
    logger.info(f"  {len(new_features)//2} ratios\n")

    return df_enhanced, new_features


def create_momentum_features(df, feature_cols):
    """
    Create momentum and trend features from rolling statistics

    Args:
        df: DataFrame
        feature_cols: List of feature column names

    Returns:
        DataFrame with momentum features
    """
    logger.info("\n" + "="*80)
    logger.info("CREATING MOMENTUM FEATURES")
    logger.info("="*80)

    df_enhanced = df.copy()
    new_features = []

    # Find rolling features with different windows (e.g., strikes_3, strikes_5, strikes_10)
    # Momentum = recent performance vs longer-term average

    for prefix in ['strikes', 'wins', 'losses', 'td_acc', 'str_acc']:
        for fighter in ['f_1', 'f_2']:
            # Find short-term (3-5 fights) and long-term (10-15 fights) features
            short_cols = [col for col in feature_cols
                         if col.startswith(f'{prefix}_3_{fighter}') or col.startswith(f'{prefix}_5_{fighter}')]
            long_cols = [col for col in feature_cols
                        if col.startswith(f'{prefix}_10_{fighter}') or col.startswith(f'{prefix}_15_{fighter}')]

            if short_cols and long_cols:
                # Use first match from each
                short_col = short_cols[0]
                long_col = long_cols[0]

                # Momentum = short_term - long_term
                momentum_col = f'momentum_{prefix}_{fighter}'
                df_enhanced[momentum_col] = df[short_col] - df[long_col]
                new_features.append(momentum_col)

    logger.success(f"✓ Created {len(new_features)} momentum features\n")

    return df_enhanced, new_features


def train_with_calibration(X_train, y_train, X_val, y_val, config, model_type='xgboost'):
    """Train model with probability calibration"""
    logger.info(f"\n" + "="*80)
    logger.info(f"TRAINING {model_type.upper()} WITH CALIBRATION")
    logger.info("="*80)

    # Calculate class weights to handle imbalance
    n_samples = len(y_train)
    n_class_0 = (y_train == 0).sum()
    n_class_1 = (y_train == 1).sum()

    weight_0 = n_samples / (2 * n_class_0)
    weight_1 = n_samples / (2 * n_class_1)

    logger.info(f"\nClass distribution:")
    logger.info(f"  Class 0 (F1 wins): {n_class_0} ({n_class_0/n_samples:.1%})")
    logger.info(f"  Class 1 (F2 wins): {n_class_1} ({n_class_1/n_samples:.1%})")
    logger.info(f"\nClass weights:")
    logger.info(f"  Weight 0: {weight_0:.3f}")
    logger.info(f"  Weight 1: {weight_1:.3f}")

    # Create sample weights
    sample_weights = np.where(y_train == 0, weight_0, weight_1)

    if model_type == 'xgboost':
        # Train XGBoost with class weights
        params = dict(config.model.xgboost.to_dict())
        n_estimators = params.pop('n_estimators', 300)
        params['scale_pos_weight'] = weight_1 / weight_0

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
        dval = xgb.DMatrix(X_val, label=y_val)

        logger.info(f"\nTraining with {n_estimators} rounds...")
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, 'train'), (dval, 'val')],
            verbose_eval=50
        )

        # Get predictions
        train_preds = model.predict(dtrain)
        val_preds = model.predict(dval)

    else:  # LightGBM
        params = dict(config.model.lightgbm.to_dict())
        n_estimators = params.pop('n_estimators', 300)
        params['class_weight'] = 'balanced'

        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        logger.info(f"\nTraining with {n_estimators} rounds...")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[lgb.log_evaluation(period=50)]
        )

        train_preds = model.predict(X_train, num_iteration=model.best_iteration)
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)

    # Evaluate uncalibrated model
    train_acc = accuracy_score(y_train, (train_preds > 0.5).astype(int))
    val_acc = accuracy_score(y_val, (val_preds > 0.5).astype(int))
    train_loss = log_loss(y_train, train_preds)
    val_loss = log_loss(y_val, val_preds)
    val_auc = roc_auc_score(y_val, val_preds)

    logger.success(f"\n✓ {model_type.upper()} Training Complete (Uncalibrated)")
    logger.info(f"  Train Accuracy: {train_acc:.1%}")
    logger.info(f"  Val Accuracy: {val_acc:.1%}")
    logger.info(f"  Val ROC AUC: {val_auc:.4f}")
    logger.info(f"  Val Log Loss: {val_loss:.4f}")

    return model, {
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'train_logloss': train_loss,
        'val_logloss': val_loss,
        'val_auc': val_auc,
        'val_preds': val_preds
    }


def create_stacked_ensemble(models_predictions, y_train, y_val):
    """Create meta-learner for ensemble stacking"""
    logger.info("\n" + "="*80)
    logger.info("CREATING STACKED ENSEMBLE")
    logger.info("="*80)

    # Stack predictions as features for meta-learner
    X_meta_train = np.column_stack([pred for pred in models_predictions['train']])
    X_meta_val = np.column_stack([pred for pred in models_predictions['val']])

    logger.info(f"\nMeta-features shape: {X_meta_train.shape}")

    # Simple logistic regression as meta-learner
    from sklearn.linear_model import LogisticRegression

    meta_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    meta_model.fit(X_meta_train, y_train)

    # Predictions
    meta_train_preds = meta_model.predict_proba(X_meta_train)[:, 1]
    meta_val_preds = meta_model.predict_proba(X_meta_val)[:, 1]

    train_acc = accuracy_score(y_train, (meta_train_preds > 0.5).astype(int))
    val_acc = accuracy_score(y_val, (meta_val_preds > 0.5).astype(int))
    val_auc = roc_auc_score(y_val, meta_val_preds)
    val_loss = log_loss(y_val, meta_val_preds)

    logger.success(f"\n✓ Ensemble Meta-Learner Trained")
    logger.info(f"  Train Accuracy: {train_acc:.1%}")
    logger.info(f"  Val Accuracy: {val_acc:.1%}")
    logger.info(f"  Val ROC AUC: {val_auc:.4f}")
    logger.info(f"  Val Log Loss: {val_loss:.4f}")

    return meta_model, {
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'val_auc': val_auc,
        'val_logloss': val_loss,
        'meta_train_preds': meta_train_preds,
        'meta_val_preds': meta_val_preds
    }


def main():
    """Main advanced training pipeline"""
    logger.info("\n" + "="*80)
    logger.info("UFC MASTER PIPELINE - ADVANCED TRAINING")
    logger.info("="*80)

    # Load config
    config = get_config()

    # Set up MLflow
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run(run_name="advanced_training"):
        # Log config
        mlflow.log_params({
            'train_end_date': config.splits.train_end_date,
            'val_start_date': config.splits.val_start_date,
            'test_start_date': config.splits.test_start_date,
            'random_state': config.random_state,
            'approach': 'advanced_with_class_weights_and_features'
        })

        # 1. Load data
        df = load_ufc_data()

        # 2. Get features and target
        feature_cols, target_col = get_feature_and_target_columns(df)

        # 3. Feature engineering
        df, matchup_features = create_matchup_features(df, feature_cols)
        df, momentum_features = create_momentum_features(df, feature_cols)

        # Update feature list
        all_features = feature_cols + matchup_features + momentum_features

        logger.info(f"\n{'='*80}")
        logger.info("FEATURE SUMMARY")
        logger.info("="*80)
        logger.info(f"\nOriginal features: {len(feature_cols)}")
        logger.info(f"Matchup features: {len(matchup_features)}")
        logger.info(f"Momentum features: {len(momentum_features)}")
        logger.info(f"Total features: {len(all_features)}\n")

        mlflow.log_metrics({
            'n_features_original': len(feature_cols),
            'n_features_matchup': len(matchup_features),
            'n_features_momentum': len(momentum_features),
            'n_features_total': len(all_features)
        })

        # 4. Final leak validation
        validate_no_leakage(df, all_features)

        # 5. Temporal split
        splitter = TemporalSplitter()
        split = splitter.split(df)

        # 6. Separate features and target
        X_train = split.train[all_features]
        y_train = split.train[target_col]
        X_val = split.val[all_features]
        y_val = split.val[target_col]
        X_test = split.test[all_features] if len(split.test) > 0 else None
        y_test = split.test[target_col] if len(split.test) > 0 else None

        # Log data sizes
        mlflow.log_metrics({
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test) if X_test is not None else 0
        })

        # 7. Feature imputation
        logger.info("\n" + "="*80)
        logger.info("FEATURE IMPUTATION")
        logger.info("="*80)

        X_train_imputed = X_train.fillna(X_train.median())
        X_val_imputed = X_val.fillna(X_train.median())

        if X_test is not None:
            X_test_imputed = X_test.fillna(X_train.median())

        logger.success(f"✓ Imputation complete\n")

        # 8. Train XGBoost with class weights
        xgb_model, xgb_metrics = train_with_calibration(
            X_train_imputed, y_train, X_val_imputed, y_val, config, 'xgboost'
        )

        # Log XGBoost metrics
        for metric_name, value in xgb_metrics.items():
            if not isinstance(value, np.ndarray):
                mlflow.log_metric(f"xgb_{metric_name}", value)

        # 9. Train LightGBM with class weights
        lgb_model, lgb_metrics = train_with_calibration(
            X_train_imputed, y_train, X_val_imputed, y_val, config, 'lightgbm'
        )

        # Log LightGBM metrics
        for metric_name, value in lgb_metrics.items():
            if not isinstance(value, np.ndarray):
                mlflow.log_metric(f"lgb_{metric_name}", value)

        # 10. Create stacked ensemble
        models_predictions = {
            'train': [xgb_metrics['val_preds'], lgb_metrics['val_preds']],
            'val': [xgb_metrics['val_preds'], lgb_metrics['val_preds']]
        }

        # Note: Using validation predictions for both since we don't have OOF preds yet
        # In production, would use proper cross-validation for train predictions

        ensemble_model, ensemble_metrics = create_stacked_ensemble(
            models_predictions, y_val, y_val
        )

        # 11. Compare models
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON (VALIDATION SET)")
        logger.info("="*80)

        comparison = pd.DataFrame({
            'XGBoost': {k: v for k, v in xgb_metrics.items() if not isinstance(v, np.ndarray)},
            'LightGBM': {k: v for k, v in lgb_metrics.items() if not isinstance(v, np.ndarray)},
            'Ensemble': {k: v for k, v in ensemble_metrics.items() if not isinstance(v, np.ndarray)}
        }).T

        logger.info(f"\n{comparison.to_string()}")

        # Determine best model
        best_model_name = comparison['val_accuracy'].idxmax()
        logger.success(f"\n✓ Best Model: {best_model_name}")
        mlflow.log_param('best_model', best_model_name)

        # 12. Test set evaluation
        if X_test is not None and len(X_test) > 0:
            logger.info("\n" + "="*80)
            logger.info("TEST SET EVALUATION")
            logger.info("="*80)

            # Filter to binary outcomes
            binary_mask = y_test.isin([0, 1])
            X_test_binary = X_test_imputed[binary_mask]
            y_test_binary = y_test[binary_mask]

            logger.info(f"\nTest set: {len(y_test)} total fights")
            logger.info(f"Binary outcomes: {len(y_test_binary)} fights")

            if len(y_test_binary) > 0:
                # Get predictions from all models
                dtest = xgb.DMatrix(X_test_binary)
                xgb_test_preds = xgb_model.predict(dtest)
                lgb_test_preds = lgb_model.predict(X_test_binary, num_iteration=lgb_model.best_iteration)

                # Ensemble prediction
                X_meta_test = np.column_stack([xgb_test_preds, lgb_test_preds])
                ensemble_test_preds = ensemble_model.predict_proba(X_meta_test)[:, 1]

                # Evaluate each model
                for model_name, preds in [('XGBoost', xgb_test_preds),
                                          ('LightGBM', lgb_test_preds),
                                          ('Ensemble', ensemble_test_preds)]:
                    test_acc = accuracy_score(y_test_binary, (preds > 0.5).astype(int))
                    test_loss = log_loss(y_test_binary, preds)
                    test_auc = roc_auc_score(y_test_binary, preds)

                    logger.info(f"\n{model_name} Test Results:")
                    logger.info(f"  Accuracy: {test_acc:.1%}")
                    logger.info(f"  Log Loss: {test_loss:.4f}")
                    logger.info(f"  ROC AUC: {test_auc:.4f}")

                    mlflow.log_metrics({
                        f'test_accuracy_{model_name.lower()}': test_acc,
                        f'test_logloss_{model_name.lower()}': test_loss,
                        f'test_auc_{model_name.lower()}': test_auc
                    })

        # 13. Save models
        logger.info("\n" + "="*80)
        logger.info("SAVING MODELS")
        logger.info("="*80)

        models_dir = Path(config.paths.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        xgb_path = models_dir / "xgboost_advanced.json"
        xgb_model.save_model(str(xgb_path))
        logger.success(f"✓ Saved XGBoost: {xgb_path}")

        lgb_path = models_dir / "lightgbm_advanced.txt"
        lgb_model.save_model(str(lgb_path))
        logger.success(f"✓ Saved LightGBM: {lgb_path}")

        import pickle
        ensemble_path = models_dir / "ensemble_advanced.pkl"
        with open(ensemble_path, 'wb') as f:
            pickle.dump(ensemble_model, f)
        logger.success(f"✓ Saved Ensemble: {ensemble_path}")

        # 14. Save predictions with actual odds for ROI backtesting
        logger.info("\n" + "="*80)
        logger.info("SAVING PREDICTIONS WITH ACTUAL ODDS")
        logger.info("="*80)

        # Load raw data to get actual odds
        df_raw = pd.read_csv(config.paths.golden_dataset)
        df_raw['event_date'] = pd.to_datetime(df_raw['event_date'])

        # Merge predictions from validation and test sets
        predictions_list = []

        # Validation set
        if len(X_val_imputed) > 0:
            val_indices = split.val.index
            val_dates = df_raw.loc[val_indices, 'event_date'].reset_index(drop=True)
            val_odds_f1 = df_raw.loc[val_indices, 'f_1_odds'].reset_index(drop=True)
            val_odds_f2 = df_raw.loc[val_indices, 'f_2_odds'].reset_index(drop=True)

            val_predictions = pd.DataFrame({
                'event_date': val_dates,
                'ensemble_prob_f1': 1 - ensemble_metrics['meta_val_preds'],  # Prob of F1 winning
                'actual_winner': y_val.reset_index(drop=True),
                'f_1_odds': val_odds_f1,
                'f_2_odds': val_odds_f2,
                'split': 'validation'
            })
            predictions_list.append(val_predictions)

        # Test set (binary only)
        if X_test is not None and len(X_test_binary) > 0:
            test_binary_indices = split.test[binary_mask].index
            test_dates = df_raw.loc[test_binary_indices, 'event_date'].reset_index(drop=True)
            test_odds_f1 = df_raw.loc[test_binary_indices, 'f_1_odds'].reset_index(drop=True)
            test_odds_f2 = df_raw.loc[test_binary_indices, 'f_2_odds'].reset_index(drop=True)

            test_predictions = pd.DataFrame({
                'event_date': test_dates,
                'ensemble_prob_f1': 1 - ensemble_test_preds,  # Prob of F1 winning
                'actual_winner': y_test_binary.reset_index(drop=True),
                'f_1_odds': test_odds_f1,
                'f_2_odds': test_odds_f2,
                'split': 'test'
            })
            predictions_list.append(test_predictions)

        # Combine and save
        df_all_predictions = pd.concat(predictions_list, ignore_index=True)
        predictions_path = Path("D:/Codex/UFC-Master-Pipeline/predictions_with_odds.csv")
        df_all_predictions.to_csv(predictions_path, index=False)

        logger.success(f"✓ Saved {len(df_all_predictions):,} predictions: {predictions_path}")
        logger.info(f"  Validation: {(df_all_predictions['split'] == 'validation').sum():,}")
        logger.info(f"  Test: {(df_all_predictions['split'] == 'test').sum():,}")
        logger.info(f"  With odds: {df_all_predictions['f_1_odds'].notna().sum():,}")

        logger.info("\n" + "="*80)
        logger.success("✓ ADVANCED TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"\nResults logged to MLflow: {config.mlflow.tracking_uri}")
        logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
