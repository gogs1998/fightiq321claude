"""
Hyperparameter Optimization using Optuna

Optimizes XGBoost and LightGBM hyperparameters for best performance.

Expected accuracy improvement: +1% to +2%
Estimated runtime: 2-6 hours (depending on trials)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from loguru import logger
import pickle
import json
from datetime import datetime

from src.data.loaders import load_ufc_data
from src.data.splitters import temporal_train_test_split
from src.data.preprocessing import FeatureTypeImputationStrategy


def objective_xgboost(trial, X_train, y_train, X_val, y_val, sample_weights_train=None):
    """
    Objective function for XGBoost optimization

    Args:
        trial: Optuna trial
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        sample_weights_train: Training sample weights

    Returns:
        Validation accuracy (to maximize)
    """
    # Suggest hyperparameters
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42,
        'tree_method': 'hist',
        'eval_metric': 'logloss'
    }

    # Train model
    model = xgb.XGBClassifier(**params)

    if sample_weights_train is not None:
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    # Predict on validation
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    logloss = log_loss(y_val, y_pred_proba)

    # Store additional metrics as trial attributes
    trial.set_user_attr('accuracy', accuracy)
    trial.set_user_attr('auc', auc)
    trial.set_user_attr('logloss', logloss)

    # Return accuracy (Optuna will maximize this)
    return accuracy


def objective_lightgbm(trial, X_train, y_train, X_val, y_val, sample_weights_train=None):
    """
    Objective function for LightGBM optimization

    Args:
        trial: Optuna trial
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        sample_weights_train: Training sample weights

    Returns:
        Validation accuracy (to maximize)
    """
    # Suggest hyperparameters
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42,
        'verbose': -1
    }

    # Train model
    model = lgb.LGBMClassifier(**params)

    if sample_weights_train is not None:
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
    else:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

    # Predict on validation
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    logloss = log_loss(y_val, y_pred_proba)

    # Store additional metrics
    trial.set_user_attr('accuracy', accuracy)
    trial.set_user_attr('auc', auc)
    trial.set_user_attr('logloss', logloss)

    return accuracy


def optimize_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weights_train: pd.Series = None,
    n_trials: int = 100,
    timeout: int = 3600
) -> optuna.Study:
    """
    Optimize hyperparameters for a given model

    Args:
        model_type: 'xgboost' or 'lightgbm'
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        sample_weights_train: Training sample weights
        n_trials: Number of trials to run
        timeout: Max time in seconds (default 1 hour)

    Returns:
        Optuna study object
    """
    logger.info("="*80)
    logger.info(f"OPTIMIZING {model_type.upper()} HYPERPARAMETERS")
    logger.info("="*80)

    # Create study
    study = optuna.create_study(
        direction='maximize',  # Maximize accuracy
        sampler=TPESampler(seed=42),
        study_name=f"{model_type}_optimization"
    )

    # Select objective function
    if model_type == 'xgboost':
        objective = lambda trial: objective_xgboost(
            trial, X_train, y_train, X_val, y_val, sample_weights_train
        )
    elif model_type == 'lightgbm':
        objective = lambda trial: objective_lightgbm(
            trial, X_train, y_train, X_val, y_val, sample_weights_train
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Run optimization
    logger.info(f"Running {n_trials} trials (timeout: {timeout}s)...")
    logger.info("This may take a while...\n")

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        n_jobs=1  # Use 1 job to avoid memory issues
    )

    # Log results
    logger.success(f"\n✓ Optimization complete!")
    logger.info(f"Best trial: #{study.best_trial.number}")
    logger.info(f"Best validation accuracy: {study.best_value:.4f}")
    logger.info(f"Best AUC: {study.best_trial.user_attrs.get('auc', 0):.4f}")
    logger.info(f"Best Log Loss: {study.best_trial.user_attrs.get('logloss', 0):.4f}")

    logger.info("\nBest hyperparameters:")
    for param, value in study.best_params.items():
        logger.info(f"  {param}: {value}")

    return study


def save_optimization_results(study: optuna.Study, model_type: str, output_dir: Path):
    """
    Save optimization results

    Args:
        study: Optuna study
        model_type: Model type name
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best parameters as JSON
    best_params_file = output_dir / f"{model_type}_best_params.json"

    results = {
        'model_type': model_type,
        'optimization_date': datetime.now().isoformat(),
        'n_trials': len(study.trials),
        'best_trial_number': study.best_trial.number,
        'best_accuracy': study.best_value,
        'best_auc': study.best_trial.user_attrs.get('auc', 0),
        'best_logloss': study.best_trial.user_attrs.get('logloss', 0),
        'best_params': study.best_params
    }

    with open(best_params_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.success(f"✓ Saved best parameters to: {best_params_file}")

    # Save study object
    study_file = output_dir / f"{model_type}_study.pkl"
    with open(study_file, 'wb') as f:
        pickle.dump(study, f)

    logger.success(f"✓ Saved study object to: {study_file}")


def main():
    """Main optimization workflow"""
    import argparse

    parser = argparse.ArgumentParser(description="Optimize XGBoost and LightGBM hyperparameters")

    parser.add_argument(
        '--model',
        choices=['xgboost', 'lightgbm', 'both'],
        default='both',
        help="Model to optimize"
    )

    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help="Number of optimization trials"
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=3600,
        help="Timeout in seconds (default: 1 hour)"
    )

    parser.add_argument(
        '--output-dir',
        default='models/optimization',
        help="Directory to save results"
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("FIGHTIQ HYPERPARAMETER OPTIMIZATION")
    logger.info("="*80 + "\n")

    # Load data
    logger.info("Loading data...")
    df = load_ufc_data()

    # Split data
    logger.info("Splitting data temporally...")
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_test_split(
        df,
        target_col='target',
        train_end_date='2024-12-31',
        val_end_date='2024-12-31',
        test_start_date='2025-01-01'
    )

    # Preprocess
    logger.info("Preprocessing features...")
    imputer = FeatureTypeImputationStrategy()
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)

    # Calculate sample weights (handle class imbalance)
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights_train = compute_sample_weight('balanced', y_train)

    logger.success("✓ Data prepared\n")
    logger.info(f"Train: {len(X_train)} fights")
    logger.info(f"Val: {len(X_val)} fights")

    # Run optimization
    output_dir = Path(args.output_dir)

    if args.model in ['xgboost', 'both']:
        xgb_study = optimize_model(
            'xgboost',
            X_train, y_train,
            X_val, y_val,
            sample_weights_train,
            n_trials=args.n_trials,
            timeout=args.timeout
        )
        save_optimization_results(xgb_study, 'xgboost', output_dir)

    if args.model in ['lightgbm', 'both']:
        lgb_study = optimize_model(
            'lightgbm',
            X_train, y_train,
            X_val, y_val,
            sample_weights_train,
            n_trials=args.n_trials,
            timeout=args.timeout
        )
        save_optimization_results(lgb_study, 'lightgbm', output_dir)

    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Review best parameters in JSON files")
    logger.info("2. Update training script with optimized hyperparameters")
    logger.info("3. Retrain models and evaluate on test set")


if __name__ == "__main__":
    main()
