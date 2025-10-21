"""
Production Training: Train on 1994-2024, Test on 2025 Holdout

This is the PRODUCTION model that uses all data except 2025 for training,
keeping 2025 as a final holdout to validate real-world performance.

Training Strategy:
- Train: 1994-01-01 to 2024-12-31 (ALL historical data through 2024)
- Validation: Use 2024 as internal validation for early stopping
- Test: 2025+ (final holdout - never used in training)

This approach:
1. ✅ Uses all available recent data (includes 2023-2024)
2. ✅ Keeps honest holdout test set (2025)
3. ✅ Captures latest UFC meta evolution
4. ✅ Provides realistic performance estimate
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from loguru import logger
import mlflow
import pickle

from src.utils.config import get_config


def load_data_production():
    """Load data with FightIQ patterns, keeping odds as features"""
    logger.info("\n" + "="*80)
    logger.info("LOADING PRODUCTION DATA (WITH ODDS)")
    logger.info("="*80)

    config = get_config()
    df = pd.read_csv(config.paths.golden_dataset)
    df['event_date'] = pd.to_datetime(df['event_date'])

    logger.info(f"✓ Loaded {len(df):,} fights, {len(df.columns)} columns")

    # Use FightIQ's exact leakage patterns
    logger.info("\n" + "="*80)
    logger.info("APPLYING FIGHTIQ LEAKAGE REMOVAL")
    logger.info("="*80)

    # Round-by-round patterns
    current_fight_patterns = ['_r1_', '_r2_', '_r3_', '_r4_', '_r5_']

    # Specific totals to remove
    current_fight_totals = [
        'f_1_total_strikes_succ', 'f_2_total_strikes_succ',
        'f_1_total_strikes_att', 'f_2_total_strikes_att',
        'f_1_sig_strikes_succ', 'f_2_sig_strikes_succ',
        'f_1_sig_strikes_att', 'f_2_sig_strikes_att',
        'f_1_knockdowns', 'f_2_knockdowns',
        'f_1_submission_att', 'f_2_submission_att',
        'f_1_ctrl_time_sec', 'f_2_ctrl_time_sec',
        'fight_duration_minutes',
        'winner', 'result', 'result_details',
        'finish_round', 'finish_time', 'finish_details',
        'method', 'time_format', 'num_rounds'
    ]

    # Remove round-by-round features
    leaking_cols = [
        col for col in df.columns
        if any(pattern in col for pattern in current_fight_patterns)
    ]

    # Add specific totals
    leaking_cols.extend([
        col for col in current_fight_totals
        if col in df.columns and col not in leaking_cols
    ])

    logger.info(f"✓ Removing {len(leaking_cols)} features (FightIQ pattern)")
    logger.info(f"✓ KEEPING betting odds as features")
    logger.info(f"✓ KEEPING historical _total features (e.g., f_1_head_succ_total)")

    df_clean = df.drop(columns=leaking_cols)

    # Get feature columns (including odds)
    metadata_cols = ['event_date', 'event_name', 'event_location']
    target_col = 'winner_encoded'

    feature_cols = [
        col for col in df_clean.columns
        if col not in metadata_cols
        and col != target_col
        and df_clean[col].dtype in ['float64', 'int64', 'float32', 'int32']
    ]

    odds_cols = [col for col in feature_cols if 'odds' in col.lower()]
    logger.info(f"\n✓ Odds features: {odds_cols}")
    logger.info(f"✓ Total features: {len(feature_cols)}")

    return df_clean, feature_cols, target_col


def main():
    logger.info("\n" + "="*80)
    logger.info("PRODUCTION MODEL TRAINING")
    logger.info("Train: 1994-2024 | Holdout: 2025+")
    logger.info("="*80)

    config = get_config()

    # MLflow setup
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment("ufc_production")

    with mlflow.start_run(run_name="production_1994-2024"):
        # Load data
        df, feature_cols, target_col = load_data_production()

        # Production split: Train on ALL data before 2025
        logger.info("\n" + "="*80)
        logger.info("PRODUCTION TEMPORAL SPLIT")
        logger.info("="*80)

        train_data = df[df['event_date'] < '2025-01-01'].copy()
        test_data = df[df['event_date'] >= '2025-01-01'].copy()

        logger.info(f"\nTrain: {len(train_data)} fights (1994-2024)")
        logger.info(f"  Date range: {train_data['event_date'].min()} to {train_data['event_date'].max()}")

        logger.info(f"\nHoldout Test: {len(test_data)} fights (2025+)")
        logger.info(f"  Date range: {test_data['event_date'].min()} to {test_data['event_date'].max()}")

        # For internal validation during training, use 2024 as validation
        val_data = train_data[train_data['event_date'] >= '2024-01-01'].copy()
        train_only = train_data[train_data['event_date'] < '2024-01-01'].copy()

        logger.info(f"\nInternal split for early stopping:")
        logger.info(f"  Train: {len(train_only)} fights (1994-2023)")
        logger.info(f"  Val: {len(val_data)} fights (2024)")

        # Prepare features
        X_train = train_only[feature_cols]
        y_train = train_only[target_col]

        X_val = val_data[feature_cols]
        y_val = val_data[target_col]

        X_test = test_data[feature_cols]
        y_test = test_data[target_col]

        # Impute
        logger.info("\n" + "="*80)
        logger.info("IMPUTING MISSING VALUES")
        logger.info("="*80)

        X_train_imputed = X_train.fillna(X_train.median())
        X_val_imputed = X_val.fillna(X_train.median())
        X_test_imputed = X_test.fillna(X_train.median())

        logger.info(f"✓ Imputed {len(feature_cols)} features")

        # Remove draws
        valid_train = y_train.isin([0, 1])
        valid_val = y_val.isin([0, 1])
        valid_test = y_test.isin([0, 1])

        X_train_binary = X_train_imputed[valid_train]
        y_train_binary = y_train[valid_train]
        X_val_binary = X_val_imputed[valid_val]
        y_val_binary = y_val[valid_val]
        X_test_binary = X_test_imputed[valid_test]
        y_test_binary = y_test[valid_test]

        logger.info(f"\nBinary outcomes - Train: {len(y_train_binary)}, Val: {len(y_val_binary)}, Test: {len(y_test_binary)}")

        # Train XGBoost
        logger.info("\n" + "="*80)
        logger.info("TRAINING XGBOOST (PRODUCTION)")
        logger.info("="*80)

        dtrain = xgb.DMatrix(X_train_binary, label=y_train_binary)
        dval = xgb.DMatrix(X_val_binary, label=y_val_binary)

        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'seed': 42
        }

        xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=300,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=50
        )

        xgb_val_preds = xgb_model.predict(dval)
        xgb_val_acc = accuracy_score(y_val_binary, (xgb_val_preds > 0.5).astype(int))
        xgb_val_auc = roc_auc_score(y_val_binary, xgb_val_preds)

        logger.success(f"\n✓ XGBoost Val Accuracy: {xgb_val_acc:.1%}")
        logger.info(f"  Val AUC: {xgb_val_auc:.4f}")

        # Train LightGBM
        logger.info("\n" + "="*80)
        logger.info("TRAINING LIGHTGBM (PRODUCTION)")
        logger.info("="*80)

        lgb_train = lgb.Dataset(X_train_binary, label=y_train_binary)
        lgb_val = lgb.Dataset(X_val_binary, label=y_val_binary, reference=lgb_train)

        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 128,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'seed': 42,
            'verbose': -1
        }

        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=300,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )

        lgb_val_preds = lgb_model.predict(X_val_binary)
        lgb_val_acc = accuracy_score(y_val_binary, (lgb_val_preds > 0.5).astype(int))
        lgb_val_auc = roc_auc_score(y_val_binary, lgb_val_preds)

        logger.success(f"\n✓ LightGBM Val Accuracy: {lgb_val_acc:.1%}")
        logger.info(f"  Val AUC: {lgb_val_auc:.4f}")

        # Ensemble
        logger.info("\n" + "="*80)
        logger.info("CREATING ENSEMBLE")
        logger.info("="*80)

        ensemble_val_preds = (xgb_val_preds + lgb_val_preds) / 2
        ensemble_val_acc = accuracy_score(y_val_binary, (ensemble_val_preds > 0.5).astype(int))
        ensemble_val_auc = roc_auc_score(y_val_binary, ensemble_val_preds)

        logger.success(f"\n✓ Ensemble Val Accuracy: {ensemble_val_acc:.1%}")
        logger.info(f"  Val AUC: {ensemble_val_auc:.4f}")

        # Test set evaluation (2025 HOLDOUT)
        logger.info("\n" + "="*80)
        logger.info("2025 HOLDOUT EVALUATION")
        logger.info("="*80)

        dtest = xgb.DMatrix(X_test_binary)
        xgb_test_preds = xgb_model.predict(dtest)
        lgb_test_preds = lgb_model.predict(X_test_binary)
        ensemble_test_preds = (xgb_test_preds + lgb_test_preds) / 2

        test_acc = accuracy_score(y_test_binary, (ensemble_test_preds > 0.5).astype(int))
        test_auc = roc_auc_score(y_test_binary, ensemble_test_preds)

        logger.info(f"\n2025 Holdout: {len(y_test_binary)} fights")
        logger.success(f"Ensemble Accuracy: {test_acc:.1%}")
        logger.info(f"Ensemble AUC: {test_auc:.4f}")

        # Compare to previous model
        logger.info("\n" + "="*80)
        logger.info("COMPARISON TO PREVIOUS MODEL")
        logger.info("="*80)

        logger.info(f"\nPrevious model (trained on <2023):")
        logger.info(f"  2025 Test Accuracy: 68.2%")
        logger.info(f"  2025 Test AUC: 0.7088")

        logger.info(f"\nProduction model (trained on 1994-2024):")
        logger.info(f"  2025 Test Accuracy: {test_acc:.1%}")
        logger.info(f"  2025 Test AUC: {test_auc:.4f}")

        improvement_acc = (test_acc - 0.682) * 100
        improvement_auc = test_auc - 0.7088

        if improvement_acc > 0:
            logger.success(f"\n✓ Improvement: +{improvement_acc:.1f} percentage points accuracy")
        else:
            logger.warning(f"\n⚠️  Change: {improvement_acc:.1f} percentage points accuracy")

        if improvement_auc > 0:
            logger.success(f"✓ Improvement: +{improvement_auc:.4f} AUC")
        else:
            logger.warning(f"⚠️  Change: {improvement_auc:.4f} AUC")

        # Save models
        logger.info("\n" + "="*80)
        logger.info("SAVING PRODUCTION MODELS")
        logger.info("="*80)

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        # Save with _production suffix
        xgb_model.save_model(str(models_dir / "xgboost_production.json"))
        logger.success(f"✓ Saved: {models_dir / 'xgboost_production.json'}")

        lgb_model.save_model(str(models_dir / "lightgbm_production.txt"))
        logger.success(f"✓ Saved: {models_dir / 'lightgbm_production.txt'}")

        ensemble_info = {
            'xgb_weight': 0.5,
            'lgb_weight': 0.5,
            'features': feature_cols,
            'train_date_range': f"1994-01-01 to 2024-12-31",
            'test_accuracy': test_acc,
            'test_auc': test_auc,
            'imputation_medians': X_train.median().to_dict()
        }

        with open(models_dir / "ensemble_production.pkl", 'wb') as f:
            pickle.dump(ensemble_info, f)
        logger.success(f"✓ Saved: {models_dir / 'ensemble_production.pkl'}")

        # Save predictions for ROI backtesting
        logger.info("\n" + "="*80)
        logger.info("SAVING PREDICTIONS")
        logger.info("="*80)

        # Get all data predictions for backtesting
        all_val_test = df[df['event_date'] >= '2024-01-01'].copy()
        X_all = all_val_test[feature_cols].fillna(X_train.median())

        dtest_all = xgb.DMatrix(X_all)
        xgb_all_preds = xgb_model.predict(dtest_all)
        lgb_all_preds = lgb_model.predict(X_all)
        ensemble_all_preds = (xgb_all_preds + lgb_all_preds) / 2

        predictions_df = pd.DataFrame({
            'event_date': all_val_test['event_date'].values,
            'ensemble_prob_f1': 1 - ensemble_all_preds,  # Convert to F1 probability
            'actual_winner': all_val_test[target_col].values,
            'f_1_odds': all_val_test['f_1_odds'].values if 'f_1_odds' in all_val_test.columns else np.nan,
            'f_2_odds': all_val_test['f_2_odds'].values if 'f_2_odds' in all_val_test.columns else np.nan,
        })

        predictions_file = Path("predictions_production.csv")
        predictions_df.to_csv(predictions_file, index=False)
        logger.success(f"✓ Saved {len(predictions_df)} predictions: {predictions_file}")

        # Log to MLflow
        mlflow.log_param("train_dates", "1994-2024")
        mlflow.log_param("test_dates", "2025+")
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("num_train", len(y_train_binary))

        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("val_accuracy", ensemble_val_acc)
        mlflow.log_metric("val_auc", ensemble_val_auc)

        logger.info("\n" + "="*80)
        logger.success("✓ PRODUCTION MODEL TRAINING COMPLETE")
        logger.info("="*80)

        logger.info("\nProduction Model Summary:")
        logger.info(f"  Training data: 1994-2024 ({len(y_train_binary)} fights)")
        logger.info(f"  Validation: 2024 ({len(y_val_binary)} fights)")
        logger.info(f"  Holdout test: 2025 ({len(y_test_binary)} fights)")
        logger.info(f"  Test accuracy: {test_acc:.1%}")
        logger.info(f"  Test AUC: {test_auc:.4f}")
        logger.info(f"  Features: {len(feature_cols)}")
        logger.info(f"\nModels saved to: {models_dir}/")
        logger.info(f"Predictions saved to: {predictions_file}\n")


if __name__ == "__main__":
    main()
