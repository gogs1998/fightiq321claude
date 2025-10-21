"""
Training WITH Betting Odds as Features

For real-world betting predictions, odds are available before the fight.
This model uses odds as features (NOT leakage) to maximize prediction accuracy.

Comparison:
- No-odds model: 61.3% test accuracy (pure statistical prediction)
- With-odds model: Expected 75-85% test accuracy (leverages market wisdom)
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
from sklearn.linear_model import LogisticRegression

from src.utils.config import get_config
from src.data.splitters import TemporalSplitter


def load_data_with_odds():
    """Load data keeping betting odds as features"""
    logger.info("\n" + "="*80)
    logger.info("LOADING DATA WITH ODDS - FIGHTIQ LEAKAGE PATTERNS")
    logger.info("="*80)

    config = get_config()
    df = pd.read_csv(config.paths.golden_dataset)
    df['event_date'] = pd.to_datetime(df['event_date'])

    logger.info(f"✓ Loaded {len(df):,} fights, {len(df.columns)} columns")

    # Use FightIQ's EXACT leakage removal approach (proven to achieve 67-69% test accuracy)
    # This is LESS aggressive than our original approach and keeps historical _total features

    logger.info("\n" + "="*80)
    logger.info("APPLYING FIGHTIQ LEAKAGE REMOVAL (67-69% TEST ACCURACY)")
    logger.info("="*80)

    # Round-by-round patterns (FightIQ loaders.py:116)
    current_fight_patterns = ['_r1_', '_r2_', '_r3_', '_r4_', '_r5_']

    # FightIQ's specific totals to remove (FightIQ loaders.py:120-128)
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
    logger.info(f"✓ KEEPING betting odds (f_1_odds, f_2_odds) as features")
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

    # Verify odds are included
    odds_cols = [col for col in feature_cols if 'odds' in col.lower()]
    logger.info(f"\n✓ Odds features included: {odds_cols}")
    logger.info(f"✓ Total features: {len(feature_cols)}")

    return df_clean, feature_cols, target_col


def main():
    logger.info("\n" + "="*80)
    logger.info("UFC MASTER PIPELINE - TRAINING WITH ODDS")
    logger.info("="*80)

    config = get_config()

    # MLflow setup
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment("ufc_with_odds")

    with mlflow.start_run(run_name="odds_inclusive_model"):
        # Load data with odds
        df, feature_cols, target_col = load_data_with_odds()

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

        if X_test is not None:
            X_test_imputed = X_test.fillna(X_train.median())

        logger.info(f"✓ Imputed features: {len(feature_cols)}")

        # Filter to binary outcomes
        binary_mask_train = y_train.isin([0, 1])
        binary_mask_val = y_val.isin([0, 1])

        X_train_binary = X_train_imputed[binary_mask_train]
        y_train_binary = y_train[binary_mask_train]

        X_val_binary = X_val_imputed[binary_mask_val]
        y_val_binary = y_val[binary_mask_val]

        logger.info(f"\nBinary outcomes - Train: {len(y_train_binary)}, Val: {len(y_val_binary)}")

        # Train XGBoost
        logger.info("\n" + "="*80)
        logger.info("TRAINING XGBOOST (WITH ODDS)")
        logger.info("="*80)

        dtrain = xgb.DMatrix(X_train_binary, label=y_train_binary)
        dval = xgb.DMatrix(X_val_binary, label=y_val_binary)

        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 5,
            'learning_rate': 0.023,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': config.random_state,
        }

        xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=300,
            evals=[(dtrain, 'train'), (dval, 'val')],
            verbose_eval=50
        )

        xgb_val_preds = xgb_model.predict(dval)
        xgb_val_acc = accuracy_score(y_val_binary, (xgb_val_preds > 0.5).astype(int))
        xgb_val_auc = roc_auc_score(y_val_binary, xgb_val_preds)
        xgb_val_loss = log_loss(y_val_binary, xgb_val_preds)

        logger.success(f"\n✓ XGBoost Val Accuracy: {xgb_val_acc:.1%}")
        logger.info(f"  Val AUC: {xgb_val_auc:.4f}")
        logger.info(f"  Val Log Loss: {xgb_val_loss:.4f}")

        # Train LightGBM
        logger.info("\n" + "="*80)
        logger.info("TRAINING LIGHTGBM (WITH ODDS)")
        logger.info("="*80)

        lgb_train = lgb.Dataset(X_train_binary, label=y_train_binary)
        lgb_val = lgb.Dataset(X_val_binary, label=y_val_binary, reference=lgb_train)

        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'max_depth': 7,
            'learning_rate': 0.023,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'random_state': config.random_state,
            'verbose': -1
        }

        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=300,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'val'],
            callbacks=[lgb.log_evaluation(50)]
        )

        lgb_val_preds = lgb_model.predict(X_val_binary, num_iteration=lgb_model.best_iteration)
        lgb_val_acc = accuracy_score(y_val_binary, (lgb_val_preds > 0.5).astype(int))
        lgb_val_auc = roc_auc_score(y_val_binary, lgb_val_preds)
        lgb_val_loss = log_loss(y_val_binary, lgb_val_preds)

        logger.success(f"\n✓ LightGBM Val Accuracy: {lgb_val_acc:.1%}")
        logger.info(f"  Val AUC: {lgb_val_auc:.4f}")
        logger.info(f"  Val Log Loss: {lgb_val_loss:.4f}")

        # Ensemble
        logger.info("\n" + "="*80)
        logger.info("CREATING ENSEMBLE")
        logger.info("="*80)

        X_meta_val = np.column_stack([xgb_val_preds, lgb_val_preds])
        ensemble_model = LogisticRegression(max_iter=1000)
        ensemble_model.fit(X_meta_val, y_val_binary)

        ensemble_val_preds = ensemble_model.predict_proba(X_meta_val)[:, 1]
        ensemble_val_acc = accuracy_score(y_val_binary, (ensemble_val_preds > 0.5).astype(int))
        ensemble_val_auc = roc_auc_score(y_val_binary, ensemble_val_preds)
        ensemble_val_loss = log_loss(y_val_binary, ensemble_val_preds)

        logger.success(f"\n✓ Ensemble Val Accuracy: {ensemble_val_acc:.1%}")
        logger.info(f"  Val AUC: {ensemble_val_auc:.4f}")
        logger.info(f"  Val Log Loss: {ensemble_val_loss:.4f}")

        # Test set evaluation
        if X_test is not None:
            logger.info("\n" + "="*80)
            logger.info("TEST SET EVALUATION")
            logger.info("="*80)

            binary_mask_test = y_test.isin([0, 1])
            X_test_binary = X_test_imputed[binary_mask_test]
            y_test_binary = y_test[binary_mask_test]

            logger.info(f"Test set: {len(y_test_binary)} binary fights")

            # XGBoost
            dtest = xgb.DMatrix(X_test_binary)
            xgb_test_preds = xgb_model.predict(dtest)
            xgb_test_acc = accuracy_score(y_test_binary, (xgb_test_preds > 0.5).astype(int))
            xgb_test_auc = roc_auc_score(y_test_binary, xgb_test_preds)

            # LightGBM
            lgb_test_preds = lgb_model.predict(X_test_binary, num_iteration=lgb_model.best_iteration)
            lgb_test_acc = accuracy_score(y_test_binary, (lgb_test_preds > 0.5).astype(int))
            lgb_test_auc = roc_auc_score(y_test_binary, lgb_test_preds)

            # Ensemble
            X_meta_test = np.column_stack([xgb_test_preds, lgb_test_preds])
            ensemble_test_preds = ensemble_model.predict_proba(X_meta_test)[:, 1]
            ensemble_test_acc = accuracy_score(y_test_binary, (ensemble_test_preds > 0.5).astype(int))
            ensemble_test_auc = roc_auc_score(y_test_binary, ensemble_test_preds)
            ensemble_test_loss = log_loss(y_test_binary, ensemble_test_preds)

            logger.info(f"\nXGBoost: {xgb_test_acc:.1%} accuracy, {xgb_test_auc:.4f} AUC")
            logger.info(f"LightGBM: {lgb_test_acc:.1%} accuracy, {lgb_test_auc:.4f} AUC")
            logger.success(f"Ensemble: {ensemble_test_acc:.1%} accuracy, {ensemble_test_auc:.4f} AUC")

            mlflow.log_metrics({
                'test_accuracy_ensemble': ensemble_test_acc,
                'test_auc_ensemble': ensemble_test_auc,
                'test_logloss_ensemble': ensemble_test_loss
            })

        # Save models
        logger.info("\n" + "="*80)
        logger.info("SAVING MODELS")
        logger.info("="*80)

        models_dir = Path(config.paths.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        xgb_path = models_dir / "xgboost_with_odds.json"
        xgb_model.save_model(str(xgb_path))
        logger.success(f"✓ Saved XGBoost: {xgb_path}")

        lgb_path = models_dir / "lightgbm_with_odds.txt"
        lgb_model.save_model(str(lgb_path))
        logger.success(f"✓ Saved LightGBM: {lgb_path}")

        import pickle
        ensemble_path = models_dir / "ensemble_with_odds.pkl"
        with open(ensemble_path, 'wb') as f:
            pickle.dump(ensemble_model, f)
        logger.success(f"✓ Saved Ensemble: {ensemble_path}")

        # Save predictions with odds
        logger.info("\n" + "="*80)
        logger.info("SAVING PREDICTIONS")
        logger.info("="*80)

        df_raw = pd.read_csv(config.paths.golden_dataset)
        df_raw['event_date'] = pd.to_datetime(df_raw['event_date'])

        predictions_list = []

        # Validation
        val_indices = split.val.index
        val_predictions = pd.DataFrame({
            'event_date': df_raw.loc[val_indices, 'event_date'].reset_index(drop=True),
            'ensemble_prob_f1': 1 - ensemble_val_preds,
            'actual_winner': y_val_binary.reset_index(drop=True),
            'f_1_odds': df_raw.loc[val_indices, 'f_1_odds'].reset_index(drop=True),
            'f_2_odds': df_raw.loc[val_indices, 'f_2_odds'].reset_index(drop=True),
            'split': 'validation'
        })
        predictions_list.append(val_predictions)

        # Test
        if X_test is not None:
            test_indices = split.test[binary_mask_test].index
            test_predictions = pd.DataFrame({
                'event_date': df_raw.loc[test_indices, 'event_date'].reset_index(drop=True),
                'ensemble_prob_f1': 1 - ensemble_test_preds,
                'actual_winner': y_test_binary.reset_index(drop=True),
                'f_1_odds': df_raw.loc[test_indices, 'f_1_odds'].reset_index(drop=True),
                'f_2_odds': df_raw.loc[test_indices, 'f_2_odds'].reset_index(drop=True),
                'split': 'test'
            })
            predictions_list.append(test_predictions)

        df_all_predictions = pd.concat(predictions_list, ignore_index=True)
        predictions_path = Path("D:/Codex/UFC-Master-Pipeline/predictions_with_odds_model.csv")
        df_all_predictions.to_csv(predictions_path, index=False)

        logger.success(f"✓ Saved {len(df_all_predictions):,} predictions: {predictions_path}")

        logger.info("\n" + "="*80)
        logger.success("✓ TRAINING WITH ODDS COMPLETE")
        logger.info("="*80)

        logger.info(f"\nComparison:")
        logger.info(f"  No-Odds Model: ~61% test accuracy")
        logger.info(f"  With-Odds Model: {ensemble_test_acc:.1%} test accuracy")
        logger.info(f"  Improvement: +{(ensemble_test_acc - 0.613) * 100:.1f} percentage points\n")


if __name__ == "__main__":
    main()
