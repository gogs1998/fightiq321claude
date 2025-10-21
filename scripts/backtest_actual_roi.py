"""
Actual ROI Backtesting with Real Historical Odds

Uses trained models WITHOUT betting odds as features,
then calculates ROI using actual historical closing odds.

This is the REAL backtest - not simulated.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import numpy as np
from loguru import logger
import pickle
import lightgbm as lgb
import xgboost as xgb

from src.utils.config import get_config


def kelly_criterion(prob, decimal_odds, fraction=0.25):
    """Calculate Kelly bet size (fractional)"""
    if decimal_odds <= 1.0 or prob <= 0 or prob >= 1:
        return 0.0
    kelly = (prob * decimal_odds - 1) / (decimal_odds - 1)
    return max(0, min(fraction, kelly))


def backtest_with_real_odds(predictions, actual_outcomes, real_odds_f1, real_odds_f2,
                            strategy='moderate', initial_bankroll=10000):
    """
    Backtest betting strategy with REAL historical odds

    Args:
        predictions: Model predicted probabilities (F1 winning)
        actual_outcomes: Actual winners (0=F1, 1=F2)
        real_odds_f1: Real historical odds for F1
        real_odds_f2: Real historical odds for F2
        strategy: 'conservative', 'moderate', or 'aggressive'
        initial_bankroll: Starting bankroll
    """

    # Strategy parameters
    if strategy == 'conservative':
        threshold = 0.60
        base_fraction = 0.02
        use_kelly = False
    elif strategy == 'moderate':
        threshold = 0.55
        base_fraction = 0.03
        use_kelly = False
    elif strategy == 'aggressive':
        threshold = 0.52
        base_fraction = 0.05
        use_kelly = True
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    bankroll = initial_bankroll
    bankroll_history = [bankroll]

    total_bets = 0
    winning_bets = 0
    total_staked = 0
    total_returned = 0

    bet_log = []

    for idx in range(len(predictions)):
        pred_prob_f1 = predictions[idx]
        pred_prob_f2 = 1 - pred_prob_f1
        actual = actual_outcomes.iloc[idx]
        odds_f1 = real_odds_f1.iloc[idx]
        odds_f2 = real_odds_f2.iloc[idx]

        # Skip if missing odds
        if pd.isna(odds_f1) or pd.isna(odds_f2):
            continue

        # Determine which fighter to bet on (higher confidence)
        if pred_prob_f1 > pred_prob_f2:
            bet_on = 0  # F1
            confidence = pred_prob_f1
            odds = odds_f1
        else:
            bet_on = 1  # F2
            confidence = pred_prob_f2
            odds = odds_f2

        # Check threshold
        if confidence < threshold:
            continue

        # Calculate bet size
        if use_kelly:
            bet_fraction = kelly_criterion(confidence, odds, fraction=0.25)
        else:
            # Scale by confidence
            confidence_scaled = (confidence - 0.5) * 2  # 0 to 1
            bet_fraction = base_fraction * (1 + confidence_scaled)

        # Limit max bet
        bet_fraction = min(bet_fraction, 0.05)
        bet_size = bet_fraction * bankroll

        if bet_size < 1:
            continue

        total_bets += 1
        total_staked += bet_size

        # Check if won
        bet_won = (bet_on == actual)

        if bet_won:
            winning_bets += 1
            payout = bet_size * odds
            profit = payout - bet_size
            total_returned += payout
            bankroll += profit
        else:
            profit = -bet_size
            bankroll -= bet_size

        bankroll_history.append(bankroll)

        bet_log.append({
            'bet_on': bet_on,
            'confidence': confidence,
            'odds': odds,
            'bet_size': bet_size,
            'won': bet_won,
            'profit': profit,
            'bankroll': bankroll
        })

    # Calculate metrics
    profit = bankroll - initial_bankroll
    roi = (profit / initial_bankroll) * 100
    win_rate = (winning_bets / total_bets * 100) if total_bets > 0 else 0
    avg_odds = total_returned / winning_bets if winning_bets > 0 else 0

    return {
        'strategy': strategy,
        'initial_bankroll': initial_bankroll,
        'final_bankroll': bankroll,
        'profit': profit,
        'roi': roi,
        'total_bets': total_bets,
        'winning_bets': winning_bets,
        'losing_bets': total_bets - winning_bets,
        'win_rate': win_rate,
        'total_staked': total_staked,
        'total_returned': total_returned,
        'avg_winning_odds': avg_odds,
        'bankroll_history': bankroll_history,
        'bet_log': bet_log
    }


def main():
    logger.info("\n" + "="*80)
    logger.info("ACTUAL ROI BACKTEST - REAL HISTORICAL ODDS")
    logger.info("="*80)

    config = get_config()

    # 1. Load prediction CSV (contains leak-free predictions + odds)
    logger.info("\n" + "="*80)
    logger.info("LOADING PREDICTIONS WITH ACTUAL ODDS")
    logger.info("="*80)

    predictions_file = Path("D:/Codex/UFC-Master-Pipeline/predictions_with_odds.csv")

    if not predictions_file.exists():
        logger.error("\n✗ predictions_with_odds.csv not found!")
        logger.info("\nPlease run train_advanced.py first to generate predictions.")
        logger.info("The training script will save predictions along with actual odds.")
        return

    df_predictions = pd.read_csv(predictions_file)
    df_predictions['event_date'] = pd.to_datetime(df_predictions['event_date'])

    logger.info(f"✓ Loaded {len(df_predictions):,} predictions")
    logger.info(f"✓ Date range: {df_predictions['event_date'].min()} to {df_predictions['event_date'].max()}")

    # Check columns
    required_cols = ['ensemble_prob_f1', 'actual_winner', 'f_1_odds', 'f_2_odds', 'event_date']
    missing_cols = [col for col in required_cols if col not in df_predictions.columns]

    if missing_cols:
        logger.error(f"\n✗ Missing columns: {missing_cols}")
        logger.info("\nAvailable columns:")
        logger.info(f"{df_predictions.columns.tolist()}")
        return

    # Check odds coverage
    odds_available = df_predictions['f_1_odds'].notna() & df_predictions['f_2_odds'].notna()
    logger.info(f"✓ Predictions with odds: {odds_available.sum():,} ({odds_available.mean():.1%})\n")

    # 2. Backtest on validation set (2023-2024)
    logger.info("="*80)
    logger.info("BACKTESTING ON VALIDATION SET (2023-2024)")
    logger.info("="*80)

    val_data = df_predictions[
        (df_predictions['event_date'] >= config.splits.val_start_date) &
        (df_predictions['event_date'] < config.splits.test_start_date)
    ].copy()

    logger.info(f"Validation predictions: {len(val_data)}")
    logger.info(f"Binary outcomes: {len(val_data)}")
    logger.info(f"Predictions with odds: {(val_data['f_1_odds'].notna()).sum()}\n")

    # 3. Backtest strategies
    logger.info("="*80)
    logger.info("BACKTESTING BETTING STRATEGIES (REAL ODDS)")
    logger.info("="*80)

    strategies = ['conservative', 'moderate', 'aggressive']
    results_all = []

    for strategy in strategies:
        logger.info(f"\n{'='*80}")
        logger.info(f"Strategy: {strategy.upper()}")
        logger.info("="*80)

        results = backtest_with_real_odds(
            val_data['ensemble_prob_f1'].values,
            val_data['actual_winner'],
            val_data['f_1_odds'],
            val_data['f_2_odds'],
            strategy=strategy,
            initial_bankroll=10000
        )

        logger.info(f"\nResults:")
        logger.info(f"  Total Bets: {results['total_bets']}")
        logger.info(f"  Winning Bets: {results['winning_bets']}")
        logger.info(f"  Losing Bets: {results['losing_bets']}")
        logger.info(f"  Win Rate: {results['win_rate']:.1f}%")
        logger.info(f"  Avg Winning Odds: {results['avg_winning_odds']:.2f}")
        logger.info(f"  Total Staked: ${results['total_staked']:,.2f}")
        logger.info(f"  Total Returned: ${results['total_returned']:,.2f}")
        logger.info(f"  Profit/Loss: ${results['profit']:,.2f}")
        logger.info(f"  ROI: {results['roi']:+.2f}%")
        logger.info(f"  Final Bankroll: ${results['final_bankroll']:,.2f}")

        if results['roi'] > 0:
            logger.success(f"\n✓ PROFITABLE! ROI: {results['roi']:+.2f}%")
        else:
            logger.error(f"\n✗ Unprofitable. ROI: {results['roi']:+.2f}%")

        results_all.append({
            'Strategy': strategy.title(),
            'Bets': results['total_bets'],
            'Win Rate': f"{results['win_rate']:.1f}%",
            'Avg Odds': f"{results['avg_winning_odds']:.2f}",
            'Profit': f"${results['profit']:,.0f}",
            'ROI': f"{results['roi']:+.1f}%",
            'Final': f"${results['final_bankroll']:,.0f}"
        })

    # 4. Summary table
    logger.info("\n" + "="*80)
    logger.info("VALIDATION SET RESULTS SUMMARY (2023-2024)")
    logger.info("="*80)

    summary_df = pd.DataFrame(results_all)
    logger.info(f"\n{summary_df.to_string(index=False)}\n")

    # 5. Test set backtest (2025)
    logger.info("="*80)
    logger.info("BACKTESTING ON TEST SET (2025 - UNSEEN)")
    logger.info("="*80)

    test_data = df_predictions[
        df_predictions['event_date'] >= config.splits.test_start_date
    ].copy()

    logger.info(f"Test predictions: {len(test_data)}")
    logger.info(f"Predictions with odds: {(test_data['f_1_odds'].notna()).sum()}\n")

    # Use best strategy from validation
    best_strategy_idx = summary_df['ROI'].apply(lambda x: float(x.strip('%+'))).argmax()
    best_strategy = summary_df.iloc[best_strategy_idx]['Strategy'].lower()

    test_results = backtest_with_real_odds(
        test_data['ensemble_prob_f1'].values,
        test_data['actual_winner'],
        test_data['f_1_odds'],
        test_data['f_2_odds'],
        strategy=best_strategy,
        initial_bankroll=10000
    )

    logger.info(f"Strategy: {best_strategy.upper()} (best from validation)\n")
    logger.info(f"Results:")
    logger.info(f"  Total Bets: {test_results['total_bets']}")
    logger.info(f"  Win Rate: {test_results['win_rate']:.1f}%")
    logger.info(f"  Avg Winning Odds: {test_results['avg_winning_odds']:.2f}")
    logger.info(f"  Profit/Loss: ${test_results['profit']:,.2f}")
    logger.info(f"  ROI: {test_results['roi']:+.2f}%")
    logger.info(f"  Final Bankroll: ${test_results['final_bankroll']:,.2f}")

    if test_results['roi'] > 0:
        logger.success(f"\n✓ PROFITABLE ON 2025 TEST SET! ROI: {test_results['roi']:+.2f}%")
    else:
        logger.warning(f"\n⚠️  Unprofitable on test set. ROI: {test_results['roi']:+.2f}%")

    # 6. Final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL BACKTEST SUMMARY (REAL HISTORICAL ODDS)")
    logger.info("="*80)

    logger.info(f"\nValidation Set (2023-2024):")
    logger.info(f"  Best Strategy: {summary_df.iloc[best_strategy_idx]['Strategy']}")
    logger.info(f"  Best ROI: {summary_df.iloc[best_strategy_idx]['ROI']}")

    logger.info(f"\nTest Set (2025):")
    logger.info(f"  Strategy: {best_strategy.title()}")
    logger.info(f"  ROI: {test_results['roi']:+.2f}%")
    logger.info(f"  Profit: ${test_results['profit']:,.2f}")

    logger.info("\n" + "="*80)
    logger.success("✓ ACTUAL ROI BACKTEST COMPLETE")
    logger.info("="*80)

    logger.info("\nKey Findings:")
    logger.info("1. Used REAL historical betting odds (not simulated)")
    logger.info("2. Model predictions made WITHOUT using odds as features")
    logger.info("3. This is the TRUE backtested performance")
    logger.info("4. Results reflect real-world betting profitability\n")


if __name__ == "__main__":
    main()
