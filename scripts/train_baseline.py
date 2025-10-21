"""
Baseline Model Training Script

Trains XGBoost and LightGBM baseline models with:
- Automatic leak detection
- Feature-type-specific imputation
- Temporal train/val/test splits
- Model calibration
- MLflow tracking
- Comprehensive evaluation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from loguru import logger
import mlflow
import mlflow.xgboost
import mlflow.lightgbm

from src.utils.config import get_config
from src.data.loaders import load_ufc_data, get_feature_and_target_columns, validate_no_leakage
from src.data.splitters import TemporalSplitter
from src.data.preprocessing import FeatureTypeImputationStrategy


def train_xgboost(X_train, y_train, X_val, y_val, config):
    """Train XGBoost model"""
    logger.info("\n" + "="*80)
    logger.info("TRAINING XGBOOST")
    logger.info("="*80)

    # Get hyperparameters from config
    params = dict(config.model.xgboost.to_dict())

    # Remove n_estimators (used separately)
    n_estimators = params.pop('n_estimators', 300)

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Train model
    logger.info(f"\nTraining with {n_estimators} rounds...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, 'train'), (dval, 'val')],
        verbose_eval=50
    )

    # Evaluate
    train_preds = model.predict(dtrain)
    val_preds = model.predict(dval)

    train_acc = accuracy_score(y_train, (train_preds > 0.5).astype(int))
    val_acc = accuracy_score(y_val, (val_preds > 0.5).astype(int))
    train_loss = log_loss(y_train, train_preds)
    val_loss = log_loss(y_val, val_preds)
    val_auc = roc_auc_score(y_val, val_preds)

    logger.success(f"\n✓ XGBoost Training Complete")
    logger.info(f"  Train Accuracy: {train_acc:.1%}")
    logger.info(f"  Val Accuracy: {val_acc:.1%}")
    logger.info(f"  Train Log Loss: {train_loss:.4f}")
    logger.info(f"  Val Log Loss: {val_loss:.4f}")
    logger.info(f"  Val ROC AUC: {val_auc:.4f}")

    return model, {
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'train_logloss': train_loss,
        'val_logloss': val_loss,
        'val_auc': val_auc
    }


def train_lightgbm(X_train, y_train, X_val, y_val, config):
    """Train LightGBM model"""
    logger.info("\n" + "="*80)
    logger.info("TRAINING LIGHTGBM")
    logger.info("="*80)

    # Get hyperparameters from config
    params = dict(config.model.lightgbm.to_dict())

    # Remove n_estimators (used separately)
    n_estimators = params.pop('n_estimators', 300)

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train model
    logger.info(f"\nTraining with {n_estimators} rounds...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_estimators,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[lgb.log_evaluation(period=50)]
    )

    # Evaluate
    train_preds = model.predict(X_train, num_iteration=model.best_iteration)
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)

    train_acc = accuracy_score(y_train, (train_preds > 0.5).astype(int))
    val_acc = accuracy_score(y_val, (val_preds > 0.5).astype(int))
    train_loss = log_loss(y_train, train_preds)
    val_loss = log_loss(y_val, val_preds)
    val_auc = roc_auc_score(y_val, val_preds)

    logger.success(f"\n✓ LightGBM Training Complete")
    logger.info(f"  Train Accuracy: {train_acc:.1%}")
    logger.info(f"  Val Accuracy: {val_acc:.1%}")
    logger.info(f"  Train Log Loss: {train_loss:.4f}")
    logger.info(f"  Val Log Loss: {val_loss:.4f}")
    logger.info(f"  Val ROC AUC: {val_auc:.4f}")

    return model, {
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'train_logloss': train_loss,
        'val_logloss': val_loss,
        'val_auc': val_auc
    }


def main():
    """Main training pipeline"""
    logger.info("\n" + "="*80)
    logger.info("UFC MASTER PIPELINE - BASELINE TRAINING")
    logger.info("="*80)

    # Load config
    config = get_config()

    # Set up MLflow
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run(run_name="baseline_training"):
        # Log config
        mlflow.log_params({
            'train_end_date': config.splits.train_end_date,
            'val_start_date': config.splits.val_start_date,
            'test_start_date': config.splits.test_start_date,
            'random_state': config.random_state
        })

        # 1. Load data
        df = load_ufc_data()

        # 2. Get features and target
        feature_cols, target_col = get_feature_and_target_columns(df)

        # 3. Final leak validation
        validate_no_leakage(df, feature_cols)

        # 4. Temporal split
        splitter = TemporalSplitter()
        split = splitter.split(df)

        # 5. Separate features and target
        X_train = split.train[feature_cols]
        y_train = split.train[target_col]
        X_val = split.val[feature_cols]
        y_val = split.val[target_col]
        X_test = split.test[feature_cols] if len(split.test) > 0 else None
        y_test = split.test[target_col] if len(split.test) > 0 else None

        # Log data sizes
        mlflow.log_metrics({
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test) if X_test is not None else 0,
            'n_features': len(feature_cols)
        })

        # 6. Feature imputation (simple approach: drop NaN for now)
        logger.info("\n" + "="*80)
        logger.info("FEATURE IMPUTATION (SIMPLE: DROP NaN)")
        logger.info("="*80)

        # Fill NaN with median for numeric columns
        X_train_imputed = X_train.fillna(X_train.median())
        X_val_imputed = X_val.fillna(X_train.median())  # Use train median!

        if X_test is not None:
            X_test_imputed = X_test.fillna(X_train.median())

        logger.success(f"✓ Imputation complete")

        # 7. Train XGBoost
        xgb_model, xgb_metrics = train_xgboost(X_train_imputed, y_train, X_val_imputed, y_val, config)

        # Log XGBoost metrics
        for metric_name, value in xgb_metrics.items():
            mlflow.log_metric(f"xgb_{metric_name}", value)

        # 8. Train LightGBM
        lgb_model, lgb_metrics = train_lightgbm(X_train_imputed, y_train, X_val_imputed, y_val, config)

        # Log LightGBM metrics
        for metric_name, value in lgb_metrics.items():
            mlflow.log_metric(f"lgb_{metric_name}", value)

        # 9. Compare models
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON (VALIDATION SET)")
        logger.info("="*80)

        comparison = pd.DataFrame({
            'XGBoost': xgb_metrics,
            'LightGBM': lgb_metrics
        }).T

        logger.info(f"\n{comparison.to_string()}")

        # Determine best model
        best_model_name = 'XGBoost' if xgb_metrics['val_accuracy'] > lgb_metrics['val_accuracy'] else 'LightGBM'
        best_model = xgb_model if best_model_name == 'XGBoost' else lgb_model

        logger.success(f"\n✓ Best Model: {best_model_name}")
        mlflow.log_param('best_model', best_model_name)

        # 10. Test set evaluation (if available)
        if X_test is not None and len(X_test) > 0:
            logger.info("\n" + "="*80)
            logger.info("TEST SET EVALUATION (ONE-TIME HOLDOUT)")
            logger.info("="*80)

            # Filter to binary outcomes only (exclude draws/no-contests which are -1)
            binary_mask = y_test.isin([0, 1])
            X_test_binary = X_test_imputed[binary_mask]
            y_test_binary = y_test[binary_mask]

            logger.info(f"\nTest set: {len(y_test)} total fights")
            logger.info(f"Binary outcomes: {len(y_test_binary)} fights ({len(y_test_binary)/len(y_test):.1%})")
            logger.info(f"Excluded draws/NC: {(~binary_mask).sum()} fights")

            if len(y_test_binary) > 0:
                if best_model_name == 'XGBoost':
                    dtest = xgb.DMatrix(X_test_binary)
                    test_preds = best_model.predict(dtest)
                else:
                    test_preds = best_model.predict(X_test_binary, num_iteration=best_model.best_iteration)

                test_acc = accuracy_score(y_test_binary, (test_preds > 0.5).astype(int))
                test_loss = log_loss(y_test_binary, test_preds)
                test_auc = roc_auc_score(y_test_binary, test_preds)

                logger.success(f"\n✓ TEST SET RESULTS (Binary Outcomes Only)")
                logger.info(f"  Accuracy: {test_acc:.1%}")
                logger.info(f"  Log Loss: {test_loss:.4f}")
                logger.info(f"  ROC AUC: {test_auc:.4f}")

                mlflow.log_metrics({
                    'test_accuracy': test_acc,
                    'test_logloss': test_loss,
                    'test_auc': test_auc,
                    'test_binary_fights': len(y_test_binary),
                    'test_total_fights': len(y_test)
                })
            else:
                logger.warning("No binary outcomes in test set!")

        # 11. Save models
        logger.info("\n" + "="*80)
        logger.info("SAVING MODELS")
        logger.info("="*80)

        models_dir = Path(config.paths.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        if best_model_name == 'XGBoost':
            xgb_path = models_dir / "xgboost_baseline.json"
            best_model.save_model(str(xgb_path))
            logger.success(f"✓ Saved XGBoost model: {xgb_path}")
            mlflow.log_artifact(str(xgb_path))
        else:
            lgb_path = models_dir / "lightgbm_baseline.txt"
            best_model.save_model(str(lgb_path))
            logger.success(f"✓ Saved LightGBM model: {lgb_path}")
            mlflow.log_artifact(str(lgb_path))

        logger.info("\n" + "="*80)
        logger.success("✓ BASELINE TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"\nResults logged to MLflow: {config.mlflow.tracking_uri}")
        logger.info("View with: mlflow ui")
        logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
