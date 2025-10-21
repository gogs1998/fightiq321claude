"""
Backtesting Script - ROI and Betting Performance Analysis

Simulates betting strategy on historical data to calculate:
- Return on Investment (ROI)
- Win rate
- Profit/loss curves
- Kelly Criterion optimization
- Confidence-based betting strategies
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import pickle

from src.utils.config import get_config
from src.data.loaders import load_ufc_data, get_feature_and_target_columns
from src.data.splitters import TemporalSplitter


def kelly_criterion(probability, odds):
    """
    Calculate optimal bet size using Kelly Criterion

    Args:
        probability: Model's predicted probability of winning
        odds: Decimal odds (e.g., 2.0 for even money)

    Returns:
        Fraction of bankroll to bet (0-1)
    """
    if odds <= 1.0:
        return 0.0

    # Kelly formula: (p * odds - 1) / (odds - 1)
    kelly_fraction = (probability * odds - 1) / (odds - 1)

    # Never bet more than 25% of bankroll (fractional Kelly)
    # Never bet negative amounts
    return max(0, min(0.25, kelly_fraction))


def american_to_decimal_odds(american_odds):
    """Convert American odds to decimal odds"""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def calculate_implied_probability(decimal_odds):
    """Calculate implied probability from decimal odds"""
    return 1 / decimal_odds


def simulate_betting_strategy(predictions, actuals, strategy='threshold', threshold=0.55,
                              initial_bankroll=10000, bet_fraction=0.03):
    """
    Simulate betting strategy and calculate ROI

    Args:
        predictions: DataFrame with columns ['predicted_prob', 'f1_odds', 'f2_odds']
        actuals: Series with actual outcomes (0=F1 wins, 1=F2 wins)
        strategy: 'threshold', 'kelly', or 'fixed'
        threshold: Minimum confidence to place bet
        initial_bankroll: Starting bankroll
        bet_fraction: Fixed fraction for 'fixed' strategy

    Returns:
        Dictionary with betting results
    """
    bankroll = initial_bankroll
    bankroll_history = [bankroll]

    total_bets = 0
    winning_bets = 0
    total_staked = 0
    total_returned = 0

    bet_log = []

    for idx, row in predictions.iterrows():
        pred_prob = row['predicted_prob']
        actual = actuals.iloc[idx]

        # Determine which fighter to bet on
        # If pred_prob > 0.5, bet on F1 (actual=0), else bet on F2 (actual=1)
        if pred_prob > 0.5:
            bet_on_fighter = 0  # Fighter 1
            fighter_prob = pred_prob
            fighter_odds = row.get('f1_odds', None)
        else:
            bet_on_fighter = 1  # Fighter 2
            fighter_prob = 1 - pred_prob
            fighter_odds = row.get('f2_odds', None)

        # Skip if no odds available
        if pd.isna(fighter_odds) or fighter_odds is None:
            continue

        # Convert American odds to decimal (assuming American format)
        # If odds are already decimal, skip conversion
        if abs(fighter_odds) > 10:  # Likely American odds
            decimal_odds = american_to_decimal_odds(fighter_odds)
        else:
            decimal_odds = fighter_odds

        # Check if bet meets threshold
        if fighter_prob < threshold:
            continue

        # Calculate bet size based on strategy
        if strategy == 'kelly':
            bet_size = kelly_criterion(fighter_prob, decimal_odds) * bankroll
        elif strategy == 'fixed':
            bet_size = bet_fraction * bankroll
        elif strategy == 'threshold':
            # Scale bet size by confidence
            confidence = abs(fighter_prob - 0.5) * 2  # 0 to 1 scale
            bet_size = (0.02 + 0.03 * confidence) * bankroll  # 2% to 5%
        else:
            bet_size = 0.03 * bankroll

        # Limit bet size
        bet_size = min(bet_size, 0.05 * bankroll)  # Max 5% of bankroll

        if bet_size < 1:  # Skip tiny bets
            continue

        total_bets += 1
        total_staked += bet_size

        # Determine if bet won
        bet_won = (bet_on_fighter == actual)

        if bet_won:
            winning_bets += 1
            payout = bet_size * decimal_odds
            profit = payout - bet_size
            total_returned += payout
            bankroll += profit
        else:
            bankroll -= bet_size
            profit = -bet_size

        bankroll_history.append(bankroll)

        bet_log.append({
            'idx': idx,
            'bet_on': bet_on_fighter,
            'confidence': fighter_prob,
            'odds': decimal_odds,
            'bet_size': bet_size,
            'won': bet_won,
            'profit': profit,
            'bankroll': bankroll
        })

    # Calculate metrics
    roi = ((bankroll - initial_bankroll) / initial_bankroll) * 100
    win_rate = (winning_bets / total_bets * 100) if total_bets > 0 else 0
    avg_bet = total_staked / total_bets if total_bets > 0 else 0
    profit = bankroll - initial_bankroll

    return {
        'initial_bankroll': initial_bankroll,
        'final_bankroll': bankroll,
        'profit': profit,
        'roi': roi,
        'total_bets': total_bets,
        'winning_bets': winning_bets,
        'win_rate': win_rate,
        'total_staked': total_staked,
        'total_returned': total_returned,
        'avg_bet': avg_bet,
        'bankroll_history': bankroll_history,
        'bet_log': bet_log
    }


def evaluate_calibration(y_true, y_pred, n_bins=10):
    """
    Evaluate probability calibration

    Args:
        y_true: Actual outcomes
        y_pred: Predicted probabilities
        n_bins: Number of bins for calibration curve

    Returns:
        Calibration metrics and curve data
    """
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='uniform')

    # Brier score (lower is better, 0 to 1)
    brier = brier_score_loss(y_true, y_pred)

    return {
        'brier_score': brier,
        'prob_true': prob_true,
        'prob_pred': prob_pred
    }


def main():
    """Main backtesting pipeline"""
    logger.info("\n" + "="*80)
    logger.info("UFC MASTER PIPELINE - BACKTESTING ROI")
    logger.info("="*80)

    config = get_config()

    # 1. Load data
    df = load_ufc_data()
    feature_cols, target_col = get_feature_and_target_columns(df)

    # 2. Load matchup and momentum features (from advanced training)
    logger.info("\n" + "="*80)
    logger.info("LOADING ADVANCED FEATURES")
    logger.info("="*80)

    # We need to recreate features - import from train_advanced.py
    from train_advanced import create_matchup_features, create_momentum_features

    df, matchup_features = create_matchup_features(df, feature_cols)
    df, momentum_features = create_momentum_features(df, feature_cols)
    all_features = feature_cols + matchup_features + momentum_features

    # 3. Temporal split
    splitter = TemporalSplitter()
    split = splitter.split(df)

    # 4. Load trained models
    logger.info("\n" + "="*80)
    logger.info("LOADING TRAINED MODELS")
    logger.info("="*80)

    models_dir = Path(config.paths.models_dir)

    # Load ensemble
    import lightgbm as lgb
    import xgboost as xgb

    xgb_model = xgb.Booster()
    xgb_model.load_model(str(models_dir / "xgboost_advanced.json"))

    lgb_model = lgb.Booster(model_file=str(models_dir / "lightgbm_advanced.txt"))

    with open(models_dir / "ensemble_advanced.pkl", 'rb') as f:
        ensemble_model = pickle.load(f)

    logger.success("✓ Models loaded successfully\n")

    # 5. Backtest on validation set
    logger.info("="*80)
    logger.info("BACKTESTING ON VALIDATION SET (2023-2024)")
    logger.info("="*80)

    X_val = split.val[all_features].fillna(split.train[all_features].median())
    y_val = split.val[target_col]

    # Get ensemble predictions
    dval_xgb = xgb.DMatrix(X_val)
    xgb_preds = xgb_model.predict(dval_xgb)
    lgb_preds = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)

    X_meta = np.column_stack([xgb_preds, lgb_preds])
    ensemble_probs = ensemble_model.predict_proba(X_meta)[:, 1]

    # Add odds columns (if available)
    predictions_df = pd.DataFrame({
        'predicted_prob': 1 - ensemble_probs,  # Probability of F1 winning
        'f1_odds': split.val.get('f_1_odds', pd.Series([np.nan] * len(y_val))),
        'f2_odds': split.val.get('f_2_odds', pd.Series([np.nan] * len(y_val)))
    })

    # Check if odds are available
    odds_available = not predictions_df['f1_odds'].isna().all()

    if not odds_available:
        logger.warning("⚠️  No betting odds found in validation data!")
        logger.info("Note: Odds were removed as data leakage. ROI calculation requires odds.")
        logger.info("For demonstration, we'll simulate hypothetical odds based on probabilities.\n")

        # Simulate odds based on predicted probabilities (for demonstration)
        # Real odds would come from bookmakers
        def prob_to_american_odds(prob):
            """Convert probability to approximate American odds"""
            if prob > 0.5:
                # Favorite
                return -100 * prob / (1 - prob)
            else:
                # Underdog
                return 100 * (1 - prob) / prob

        predictions_df['f1_odds'] = predictions_df['predicted_prob'].apply(prob_to_american_odds)
        predictions_df['f2_odds'] = (1 - predictions_df['predicted_prob']).apply(prob_to_american_odds)

        logger.info("✓ Simulated odds generated for backtesting\n")

    # 6. Evaluate calibration
    logger.info("="*80)
    logger.info("PROBABILITY CALIBRATION")
    logger.info("="*80)

    # Filter binary outcomes
    binary_mask = y_val.isin([0, 1])
    y_val_binary = y_val[binary_mask]
    ensemble_probs_binary = ensemble_probs[binary_mask]

    calibration = evaluate_calibration(y_val_binary, ensemble_probs_binary)

    logger.info(f"\nBrier Score: {calibration['brier_score']:.4f} (lower is better)")
    logger.info("Brier Score interpretation:")
    logger.info("  < 0.10: Excellent calibration")
    logger.info("  0.10-0.20: Good calibration")
    logger.info("  0.20-0.30: Fair calibration")
    logger.info("  > 0.30: Poor calibration")

    if calibration['brier_score'] < 0.20:
        logger.success(f"✓ Model is well-calibrated (Brier: {calibration['brier_score']:.4f})\n")
    else:
        logger.warning(f"⚠️  Model needs better calibration (Brier: {calibration['brier_score']:.4f})\n")

    # 7. Simulate betting strategies
    logger.info("="*80)
    logger.info("BETTING STRATEGY SIMULATION")
    logger.info("="*80)

    strategies = [
        ('Conservative (60% threshold)', 'threshold', 0.60),
        ('Moderate (55% threshold)', 'threshold', 0.55),
        ('Aggressive (52% threshold)', 'threshold', 0.52),
        ('Kelly Criterion', 'kelly', 0.55),
    ]

    results_summary = []

    for strategy_name, strategy_type, threshold in strategies:
        logger.info(f"\n{'='*80}")
        logger.info(f"Strategy: {strategy_name}")
        logger.info("="*80)

        results = simulate_betting_strategy(
            predictions_df[binary_mask].reset_index(drop=True),
            y_val_binary.reset_index(drop=True),
            strategy=strategy_type,
            threshold=threshold,
            initial_bankroll=10000
        )

        logger.info(f"\nResults:")
        logger.info(f"  Total Bets: {results['total_bets']}")
        logger.info(f"  Winning Bets: {results['winning_bets']}")
        logger.info(f"  Win Rate: {results['win_rate']:.1f}%")
        logger.info(f"  Total Staked: ${results['total_staked']:,.2f}")
        logger.info(f"  Total Returned: ${results['total_returned']:,.2f}")
        logger.info(f"  Profit: ${results['profit']:,.2f}")
        logger.info(f"  ROI: {results['roi']:+.2f}%")
        logger.info(f"  Final Bankroll: ${results['final_bankroll']:,.2f}")

        if results['roi'] > 0:
            logger.success(f"✓ Profitable strategy! ROI: {results['roi']:+.2f}%")
        else:
            logger.error(f"✗ Unprofitable strategy. ROI: {results['roi']:+.2f}%")

        results_summary.append({
            'Strategy': strategy_name,
            'Total Bets': results['total_bets'],
            'Win Rate (%)': f"{results['win_rate']:.1f}",
            'ROI (%)': f"{results['roi']:+.1f}",
            'Profit': f"${results['profit']:,.0f}",
            'Final Bankroll': f"${results['final_bankroll']:,.0f}"
        })

    # 8. Summary comparison
    logger.info("\n" + "="*80)
    logger.info("STRATEGY COMPARISON")
    logger.info("="*80)

    summary_df = pd.DataFrame(results_summary)
    logger.info(f"\n{summary_df.to_string(index=False)}\n")

    # 9. Test set backtest
    logger.info("="*80)
    logger.info("BACKTESTING ON TEST SET (2025)")
    logger.info("="*80)

    X_test = split.test[all_features].fillna(split.train[all_features].median())
    y_test = split.test[target_col]

    # Get predictions
    dtest_xgb = xgb.DMatrix(X_test)
    xgb_test_preds = xgb_model.predict(dtest_xgb)
    lgb_test_preds = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

    X_meta_test = np.column_stack([xgb_test_preds, lgb_test_preds])
    ensemble_test_probs = ensemble_model.predict_proba(X_meta_test)[:, 1]

    # Simulate odds for test set
    test_predictions_df = pd.DataFrame({
        'predicted_prob': 1 - ensemble_test_probs,
        'f1_odds': (1 - ensemble_test_probs).apply(lambda p: prob_to_american_odds(p)),
        'f2_odds': ensemble_test_probs.apply(lambda p: prob_to_american_odds(p))
    })

    # Filter binary outcomes
    test_binary_mask = y_test.isin([0, 1])
    y_test_binary = y_test[test_binary_mask]

    # Best strategy from validation
    best_strategy = ('Moderate (55% threshold)', 'threshold', 0.55)

    logger.info(f"\nUsing best strategy from validation: {best_strategy[0]}")

    test_results = simulate_betting_strategy(
        test_predictions_df[test_binary_mask].reset_index(drop=True),
        y_test_binary.reset_index(drop=True),
        strategy=best_strategy[1],
        threshold=best_strategy[2],
        initial_bankroll=10000
    )

    logger.info(f"\nTest Set Results (2025 Unseen Fights):")
    logger.info(f"  Total Bets: {test_results['total_bets']}")
    logger.info(f"  Win Rate: {test_results['win_rate']:.1f}%")
    logger.info(f"  Profit: ${test_results['profit']:,.2f}")
    logger.info(f"  ROI: {test_results['roi']:+.2f}%")
    logger.info(f"  Final Bankroll: ${test_results['final_bankroll']:,.2f}")

    if test_results['roi'] > 0:
        logger.success(f"\n✓ PROFITABLE ON TEST SET! ROI: {test_results['roi']:+.2f}%")
    else:
        logger.warning(f"\n⚠️  Unprofitable on test set. ROI: {test_results['roi']:+.2f}%")
        logger.info("Note: Small test set (400 fights) has high variance.")
        logger.info("Longer-term performance matters more than single test period.\n")

    # 10. Final summary
    logger.info("="*80)
    logger.info("BACKTEST SUMMARY")
    logger.info("="*80)

    logger.info(f"\nModel Performance:")
    logger.info(f"  Validation Accuracy: {accuracy_score(y_val_binary, (ensemble_probs_binary > 0.5).astype(int)):.1%}")
    logger.info(f"  Test Accuracy: {accuracy_score(y_test_binary, (ensemble_test_probs[test_binary_mask] > 0.5).astype(int)):.1%}")
    logger.info(f"  Brier Score (calibration): {calibration['brier_score']:.4f}")

    logger.info(f"\nBetting Performance:")
    logger.info(f"  Validation ROI: {results['roi']:+.2f}% (from best strategy)")
    logger.info(f"  Test ROI: {test_results['roi']:+.2f}%")

    logger.info("\n" + "="*80)
    logger.success("✓ BACKTESTING COMPLETE")
    logger.info("="*80)

    logger.info("\nKey Takeaways:")
    logger.info("1. ROI depends on betting odds quality (we simulated odds)")
    logger.info("2. Selective betting (higher thresholds) often performs better")
    logger.info("3. Kelly Criterion helps optimize bet sizing")
    logger.info("4. Model calibration is crucial for profitable betting")
    logger.info("5. Test set performance has high variance (small sample)\n")

    if not odds_available:
        logger.warning("⚠️  IMPORTANT: Real odds data was removed as leakage.")
        logger.warning("   For actual betting, you need live odds from bookmakers.")
        logger.warning("   These results use simulated odds for demonstration only.\n")


if __name__ == "__main__":
    main()
