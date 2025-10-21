"""
Training WITH Betting Odds - Using FightIQ's EXACT Leakage Patterns

This uses FightIQ's exact leakage removal approach to see if we can
match their 69% test accuracy.
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

from src.utils.config import get_config
from src.data.splitters import TemporalSplitter


def load_data_fightiq_style():
    """Load data using FightIQ's EXACT leakage removal patterns"""
    logger.info("\n" + "="*80)
    logger.info("LOADING DATA - FIGHT IQ LEAKAGE PATTERNS")
    logger.info("="*80)

    config = get_config()
    df = pd.read_csv(config.paths.golden_dataset)
    df['event_date'] = pd.to_datetime(df['event_date'])

    logger.info(f"✓ Loaded {len(df):,} fights, {len(df.columns)} columns")

    # FightIQ's EXACT leakage patterns (from loaders.py:115-128)
    logger.info("\n" + "="*80)
    logger.info("APPLYING FIGHTIQ LEAKAGE REMOVAL")
    logger.info("="*80)

    # Round-by-round patterns (FightIQ line 116)
    current_fight_patterns = ['_r1_', '_r2_', '_r3_', '_r4_', '_r5_']

    # FightIQ's specific totals (FightIQ lines 120-128)
    current_fight_totals = [
        'f_1_total_strikes_succ', 'f_2_total_strikes_succ',
        'f_1_total_strikes_att', 'f_2_total_strikes_att',
        'f_1_sig_strikes_succ', 'f_2_sig_strikes_succ',
        'f_1_sig_strikes_att', 'f_2_sig_strikes_att',
        'f_1_knockdowns', 'f_2_knockdowns',
        'f_1_submission_att', 'f_2_submission_att',
        'f_1_ctrl_time_sec', 'f_2_ctrl_time_sec',
        'fight_duration_minutes'
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

    df_clean = df.drop(columns=leaking_cols)

    # Get feature columns (same as FightIQ)
    metadata_cols = ['event_date', 'event_name', 'event_location']
    target_col = 'winner_encoded'

    feature_cols = [
        col for col in df_clean.columns
        if col not in metadata_cols
        and col != target_col
        and df_clean[col].dtype in ['float64', 'int64', 'float32', 'int32']
    ]

    # Verify odds are included
    odds_cols = [col for col in feature_cols if 'odds' in col.lower()]
    logger.info(f"\n✓ Odds features included: {odds_cols}")
    logger.info(f"✓ Total features: {len(feature_cols)}")

    # Count how many _total features we're KEEPING (that our old model removed)
    kept_totals = [col for col in feature_cols if '_total' in col]
    logger.info(f"✓ Features with '_total' kept: {len(kept_totals)}")
    logger.info(f"  Sample: {kept_totals[:10]}")

    return df_clean, feature_cols, target_col


def main():
    logger.info("\n" + "="*80)
    logger.info("FIGHTIQ PATTERN TRAINING")
    logger.info("="*80)

    config = get_config()

    # Load data with FightIQ patterns
    df, feature_cols, target_col = load_data_fightiq_style()

    # Temporal split
    logger.info("\n" + "="*80)
    logger.info("TEMPORAL SPLIT")
    logger.info("="*80)

    splitter = TemporalSplitter()
    split = splitter.split(df)

    # Prepare data
    X_train = split.train[feature_cols]
    y_train = split.train[target_col]

    X_val = split.val[feature_cols]
    y_val = split.val[target_col]

    X_test = split.test[feature_cols] if len(split.test) > 0 else None
    y_test = split.test[target_col] if len(split.test) > 0 else None

    # Impute missing values
    logger.info("\n" + "="*80)
    logger.info("IMPUTING MISSING VALUES")
    logger.info("="*80)

    X_train_imputed = X_train.fillna(X_train.median())
    X_val_imputed = X_val.fillna(X_train.median())
    X_test_imputed = X_test.fillna(X_train.median()) if X_test is not None else None

    logger.info(f"✓ Imputed features: {len(feature_cols)}")

    # Remove draws (binary classification)
    valid_train = y_train.isin([0, 1])
    valid_val = y_val.isin([0, 1])

    X_train_binary = X_train_imputed[valid_train]
    y_train_binary = y_train[valid_train]
    X_val_binary = X_val_imputed[valid_val]
    y_val_binary = y_val[valid_val]

    logger.info(f"\nBinary outcomes - Train: {len(y_train_binary)}, Val: {len(y_val_binary)}")

    # Train XGBoost
    logger.info("\n" + "="*80)
    logger.info("TRAINING XGBOOST (FIGHTIQ PATTERNS)")
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
    xgb_val_logloss = log_loss(y_val_binary, xgb_val_preds)

    logger.success(f"\n✓ XGBoost Val Accuracy: {xgb_val_acc:.1%}")
    logger.info(f"  Val AUC: {xgb_val_auc:.4f}")
    logger.info(f"  Val Log Loss: {xgb_val_logloss:.4f}")

    # Train LightGBM
    logger.info("\n" + "="*80)
    logger.info("TRAINING LIGHTGBM (FIGHTIQ PATTERNS)")
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
    lgb_val_logloss = log_loss(y_val_binary, lgb_val_preds)

    logger.success(f"\n✓ LightGBM Val Accuracy: {lgb_val_acc:.1%}")
    logger.info(f"  Val AUC: {lgb_val_auc:.4f}")
    logger.info(f"  Val Log Loss: {lgb_val_logloss:.4f}")

    # Ensemble
    logger.info("\n" + "="*80)
    logger.info("CREATING ENSEMBLE")
    logger.info("="*80)

    ensemble_val_preds = (xgb_val_preds + lgb_val_preds) / 2
    ensemble_val_acc = accuracy_score(y_val_binary, (ensemble_val_preds > 0.5).astype(int))
    ensemble_val_auc = roc_auc_score(y_val_binary, ensemble_val_preds)
    ensemble_val_logloss = log_loss(y_val_binary, ensemble_val_preds)

    logger.success(f"\n✓ Ensemble Val Accuracy: {ensemble_val_acc:.1%}")
    logger.info(f"  Val AUC: {ensemble_val_auc:.4f}")
    logger.info(f"  Val Log Loss: {ensemble_val_logloss:.4f}")

    # Test set evaluation
    if X_test_imputed is not None:
        logger.info("\n" + "="*80)
        logger.info("TEST SET EVALUATION")
        logger.info("="*80)

        valid_test = y_test.isin([0, 1])
        X_test_binary = X_test_imputed[valid_test]
        y_test_binary = y_test[valid_test]

        logger.info(f"Test set: {len(y_test_binary)} binary fights")

        # XGBoost test
        dtest = xgb.DMatrix(X_test_binary)
        xgb_test_preds = xgb_model.predict(dtest)
        xgb_test_acc = accuracy_score(y_test_binary, (xgb_test_preds > 0.5).astype(int))
        xgb_test_auc = roc_auc_score(y_test_binary, xgb_test_preds)

        # LightGBM test
        lgb_test_preds = lgb_model.predict(X_test_binary)
        lgb_test_acc = accuracy_score(y_test_binary, (lgb_test_preds > 0.5).astype(int))
        lgb_test_auc = roc_auc_score(y_test_binary, lgb_test_preds)

        # Ensemble test
        ensemble_test_preds = (xgb_test_preds + lgb_test_preds) / 2
        ensemble_test_acc = accuracy_score(y_test_binary, (ensemble_test_preds > 0.5).astype(int))
        ensemble_test_auc = roc_auc_score(y_test_binary, ensemble_test_preds)

        logger.info(f"\nXGBoost: {xgb_test_acc:.1%} accuracy, {xgb_test_auc:.4f} AUC")
        logger.info(f"LightGBM: {lgb_test_acc:.1%} accuracy, {lgb_test_auc:.4f} AUC")
        logger.success(f"Ensemble: {ensemble_test_acc:.1%} accuracy, {ensemble_test_auc:.4f} AUC")

        # Compare to FightIQ's 69%
        logger.info("\n" + "="*80)
        logger.info("COMPARISON TO FIGHTIQ")
        logger.info("="*80)
        logger.info(f"\nFightIQ Test Accuracy: 69.0%")
        logger.info(f"Our Test Accuracy: {ensemble_test_acc:.1%}")
        logger.info(f"Difference: {(ensemble_test_acc - 0.69):.1%} percentage points")

        if ensemble_test_acc >= 0.68:
            logger.success("\n✓ MATCHED FIGHTIQ PERFORMANCE!")
        else:
            logger.warning(f"\n⚠️  Still {(0.69 - ensemble_test_acc):.1%} below FightIQ")

    logger.info("\n" + "="*80)
    logger.success("✓ FIGHTIQ PATTERN TRAINING COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
