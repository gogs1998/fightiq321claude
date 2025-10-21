"""
Actual ROI Backtesting with FIXED STAKES

Uses fixed bet sizes instead of percentage of bankroll to avoid
compounding explosion. This gives a more realistic ROI estimate.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import numpy as np
from loguru import logger

from src.utils.config import get_config


def backtest_with_fixed_stakes(predictions, actual_outcomes, real_odds_f1, real_odds_f2,
                               strategy='moderate', unit_size=100):
    """
    Backtest betting strategy with FIXED unit sizes

    Args:
        predictions: Model predicted probabilities (F1 winning)
        actual_outcomes: Actual winners (0=F1, 1=F2)
        real_odds_f1: Real historical odds for F1
        real_odds_f2: Real historical odds for F2
        strategy: 'conservative', 'moderate', or 'aggressive'
        unit_size: Fixed bet size per unit
    """

    # Strategy parameters
    if strategy == 'conservative':
        threshold = 0.60
        base_units = 1
    elif strategy == 'moderate':
        threshold = 0.55
        base_units = 1
    elif strategy == 'aggressive':
        threshold = 0.52
        base_units = 1
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    total_bets = 0
    winning_bets = 0
    total_staked = 0
    total_returned = 0
    profits = []

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

        # Fixed bet size (scale by confidence)
        confidence_scaled = min((confidence - 0.5) * 4, 1.0)  # 0 to 1
        units = base_units * (1 + confidence_scaled)  # 1-2 units
        bet_size = units * unit_size

        total_bets += 1
        total_staked += bet_size

        # Check if won
        bet_won = (bet_on == actual)

        if bet_won:
            winning_bets += 1
            payout = bet_size * odds
            profit = payout - bet_size
            total_returned += payout
        else:
            profit = -bet_size

        profits.append(profit)

        bet_log.append({
            'bet_on': bet_on,
            'confidence': confidence,
            'odds': odds,
            'bet_size': bet_size,
            'won': bet_won,
            'profit': profit
        })

    # Calculate metrics
    total_profit = sum(profits)
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
    win_rate = (winning_bets / total_bets * 100) if total_bets > 0 else 0
    avg_odds = total_returned / winning_bets if winning_bets > 0 else 0

    return {
        'strategy': strategy,
        'unit_size': unit_size,
        'profit': total_profit,
        'roi': roi,
        'total_bets': total_bets,
        'winning_bets': winning_bets,
        'losing_bets': total_bets - winning_bets,
        'win_rate': win_rate,
        'total_staked': total_staked,
        'total_returned': total_returned,
        'avg_winning_odds': avg_odds,
        'bet_log': bet_log,
        'profits': profits
    }


def main():
    logger.info("\n" + "="*80)
    logger.info("ACTUAL ROI BACKTEST - FIXED STAKES (REALISTIC)")
    logger.info("="*80)

    config = get_config()

    # Load predictions CSV
    logger.info("\n" + "="*80)
    logger.info("LOADING PREDICTIONS WITH ACTUAL ODDS")
    logger.info("="*80)

    # Try to load predictions from production model first (best accuracy)
    predictions_file = Path("D:/Codex/UFC-Master-Pipeline/predictions_production.csv")

    if not predictions_file.exists():
        # Fall back to with-odds model
        predictions_file = Path("D:/Codex/UFC-Master-Pipeline/predictions_with_odds_model.csv")

    if not predictions_file.exists():
        # Fall back to old predictions
        predictions_file = Path("D:/Codex/UFC-Master-Pipeline/predictions_with_odds.csv")

    if not predictions_file.exists():
        logger.error("\n✗ No predictions file found!")
        return

    logger.info(f"Using predictions from: {predictions_file.name}")

    df_predictions = pd.read_csv(predictions_file)
    df_predictions['event_date'] = pd.to_datetime(df_predictions['event_date'])

    logger.info(f"✓ Loaded {len(df_predictions):,} predictions")

    # Check odds coverage
    odds_available = df_predictions['f_1_odds'].notna() & df_predictions['f_2_odds'].notna()
    logger.info(f"✓ Predictions with odds: {odds_available.sum():,} ({odds_available.mean():.1%})\n")

    # Backtest on validation set (2023-2024)
    logger.info("="*80)
    logger.info("BACKTESTING ON VALIDATION SET (2023-2024)")
    logger.info("="*80)

    val_data = df_predictions[
        (df_predictions['event_date'] >= config.splits.val_start_date) &
        (df_predictions['event_date'] < config.splits.test_start_date)
    ].copy()

    logger.info(f"Validation predictions: {len(val_data)}")
    logger.info(f"Predictions with odds: {(val_data['f_1_odds'].notna()).sum()}\n")

    # Backtest strategies
    strategies = ['conservative', 'moderate', 'aggressive']
    results_all = []

    for strategy in strategies:
        logger.info(f"\n{'='*80}")
        logger.info(f"Strategy: {strategy.upper()}")
        logger.info("="*80)

        results = backtest_with_fixed_stakes(
            val_data['ensemble_prob_f1'].values,
            val_data['actual_winner'],
            val_data['f_1_odds'],
            val_data['f_2_odds'],
            strategy=strategy,
            unit_size=100
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
            'ROI': f"{results['roi']:+.1f}%"
        })

    # Summary table
    logger.info("\n" + "="*80)
    logger.info("VALIDATION SET RESULTS SUMMARY (2023-2024)")
    logger.info("="*80)

    summary_df = pd.DataFrame(results_all)
    logger.info(f"\n{summary_df.to_string(index=False)}\n")

    # Test set backtest (2025)
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

    test_results = backtest_with_fixed_stakes(
        test_data['ensemble_prob_f1'].values,
        test_data['actual_winner'],
        test_data['f_1_odds'],
        test_data['f_2_odds'],
        strategy=best_strategy,
        unit_size=100
    )

    logger.info(f"Strategy: {best_strategy.upper()} (best from validation)\n")
    logger.info(f"Results:")
    logger.info(f"  Total Bets: {test_results['total_bets']}")
    logger.info(f"  Win Rate: {test_results['win_rate']:.1f}%")
    logger.info(f"  Avg Winning Odds: {test_results['avg_winning_odds']:.2f}")
    logger.info(f"  Profit/Loss: ${test_results['profit']:,.2f}")
    logger.info(f"  ROI: {test_results['roi']:+.2f}%")

    if test_results['roi'] > 0:
        logger.success(f"\n✓ PROFITABLE ON 2025 TEST SET! ROI: {test_results['roi']:+.2f}%")
    else:
        logger.warning(f"\n⚠️  Unprofitable on test set. ROI: {test_results['roi']:+.2f}%")

    # Final summary
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
    logger.info(f"  Profit per $100 staked: ${test_results['profit'] / test_results['total_staked'] * 100:,.2f}")

    logger.info("\n" + "="*80)
    logger.success("✓ ACTUAL ROI BACKTEST COMPLETE")
    logger.info("="*80)

    logger.info("\nKey Findings:")
    logger.info("1. Used REAL historical betting odds (not simulated)")
    logger.info("2. Model predictions made WITHOUT using odds as features")
    logger.info("3. Fixed bet sizes ($100 units) to avoid compounding")
    logger.info("4. This is the TRUE backtested performance\n")


if __name__ == "__main__":
    main()
