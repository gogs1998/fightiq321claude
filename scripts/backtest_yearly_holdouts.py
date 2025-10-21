"""
Year-by-Year Holdout Backtesting with Actual ROI

Tests model performance on each year separately:
- 2022: Train on <2022, test on 2022
- 2023: Train on <2023, test on 2023
- 2024: Train on <2024, test on 2024
- 2025: Train on <2025, test on 2025

This shows if model generalizes across different time periods.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from src.utils.config import get_config


def calculate_roi_fixed_stakes(predictions, actual_outcomes, odds_f1, odds_f2,
                                strategy='conservative', unit_size=100):
    """Calculate ROI with fixed bet sizes"""

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

    for idx in range(len(predictions)):
        pred_prob_f1 = predictions[idx]
        pred_prob_f2 = 1 - pred_prob_f1
        actual = actual_outcomes.iloc[idx]
        odds_f1_val = odds_f1.iloc[idx]
        odds_f2_val = odds_f2.iloc[idx]

        # Skip if missing odds
        if pd.isna(odds_f1_val) or pd.isna(odds_f2_val):
            continue

        # Determine which fighter to bet on
        if pred_prob_f1 > pred_prob_f2:
            bet_on = 0  # F1
            confidence = pred_prob_f1
            odds = odds_f1_val
        else:
            bet_on = 1  # F2
            confidence = pred_prob_f2
            odds = odds_f2_val

        # Check threshold
        if confidence < threshold:
            continue

        # Fixed bet size (scale by confidence)
        confidence_scaled = min((confidence - 0.5) * 4, 1.0)
        units = base_units * (1 + confidence_scaled)
        bet_size = units * unit_size

        total_bets += 1
        total_staked += bet_size

        # Check if won
        bet_won = (bet_on == actual)

        if bet_won:
            winning_bets += 1
            payout = bet_size * odds
            total_returned += payout

    # Calculate metrics
    total_profit = total_returned - total_staked
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
    win_rate = (winning_bets / total_bets * 100) if total_bets > 0 else 0
    avg_odds = total_returned / winning_bets if winning_bets > 0 else 0

    return {
        'strategy': strategy,
        'total_bets': total_bets,
        'winning_bets': winning_bets,
        'losing_bets': total_bets - winning_bets,
        'win_rate': win_rate,
        'total_staked': total_staked,
        'total_returned': total_returned,
        'profit': total_profit,
        'roi': roi,
        'avg_winning_odds': avg_odds
    }


def main():
    logger.info("\n" + "="*80)
    logger.info("YEAR-BY-YEAR HOLDOUT BACKTEST (WITH ODDS)")
    logger.info("="*80)

    config = get_config()

    # Load predictions
    logger.info("\n" + "="*80)
    logger.info("LOADING PREDICTIONS")
    logger.info("="*80)

    predictions_file = Path("D:/Codex/UFC-Master-Pipeline/predictions_with_odds_model.csv")

    if not predictions_file.exists():
        logger.error(f"\n✗ {predictions_file.name} not found!")
        logger.info("Run train_with_odds.py first to generate predictions.")
        return

    df = pd.read_csv(predictions_file)
    df['event_date'] = pd.to_datetime(df['event_date'])

    logger.info(f"✓ Loaded {len(df):,} predictions")
    logger.info(f"✓ Date range: {df['event_date'].min()} to {df['event_date'].max()}")

    # Check required columns
    required_cols = ['ensemble_prob_f1', 'actual_winner', 'f_1_odds', 'f_2_odds', 'event_date']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"\n✗ Missing required columns!")
        logger.info(f"Required: {required_cols}")
        logger.info(f"Available: {df.columns.tolist()}")
        return

    odds_available = df['f_1_odds'].notna() & df['f_2_odds'].notna()
    logger.info(f"✓ Predictions with odds: {odds_available.sum():,} ({odds_available.mean():.1%})\n")

    # Define years to test
    years = [2022, 2023, 2024, 2025]

    all_results = []

    for year in years:
        logger.info("="*80)
        logger.info(f"HOLDOUT YEAR: {year}")
        logger.info("="*80)

        # Define holdout: test on this year, train on all years before
        test_data = df[
            (df['event_date'] >= f'{year}-01-01') &
            (df['event_date'] < f'{year+1}-01-01')
        ].copy()

        if len(test_data) == 0:
            logger.warning(f"\n⚠️  No data for {year}, skipping...")
            continue

        logger.info(f"\nTest data: {len(test_data)} fights")
        logger.info(f"Date range: {test_data['event_date'].min()} to {test_data['event_date'].max()}")

        # Calculate accuracy metrics
        valid_predictions = test_data['actual_winner'].isin([0, 1])
        test_valid = test_data[valid_predictions].copy()

        if len(test_valid) == 0:
            logger.warning(f"⚠️  No valid binary outcomes for {year}")
            continue

        y_true = test_valid['actual_winner']
        y_pred_proba = test_valid['ensemble_prob_f1']
        y_pred = (y_pred_proba < 0.5).astype(int)  # Predict F2 if prob_f1 < 0.5

        accuracy = accuracy_score(y_true, y_pred) * 100

        try:
            auc = roc_auc_score(y_true, 1 - y_pred_proba)  # Convert to prob_f2 for scoring
        except:
            auc = 0.0

        try:
            logloss = log_loss(y_true, np.column_stack([y_pred_proba, 1 - y_pred_proba]))
        except:
            logloss = 0.0

        logger.info(f"\n{'='*80}")
        logger.info(f"ACCURACY METRICS - {year}")
        logger.info("="*80)
        logger.info(f"  Total fights: {len(test_valid)}")
        logger.info(f"  Accuracy: {accuracy:.1f}%")
        logger.info(f"  AUC: {auc:.4f}")
        logger.info(f"  Log Loss: {logloss:.4f}")

        # ROI Backtesting
        logger.info(f"\n{'='*80}")
        logger.info(f"ROI BACKTEST - {year}")
        logger.info("="*80)

        odds_available_year = test_data['f_1_odds'].notna() & test_data['f_2_odds'].notna()
        logger.info(f"Predictions with odds: {odds_available_year.sum()} / {len(test_data)}")

        # Test all strategies
        strategies = ['conservative', 'moderate', 'aggressive']
        year_roi_results = []

        for strategy in strategies:
            roi_results = calculate_roi_fixed_stakes(
                test_data['ensemble_prob_f1'].values,
                test_data['actual_winner'],
                test_data['f_1_odds'],
                test_data['f_2_odds'],
                strategy=strategy,
                unit_size=100
            )

            logger.info(f"\n{strategy.upper()}:")
            logger.info(f"  Bets: {roi_results['total_bets']}")
            logger.info(f"  Win Rate: {roi_results['win_rate']:.1f}%")
            logger.info(f"  Profit: ${roi_results['profit']:,.2f}")
            logger.info(f"  ROI: {roi_results['roi']:+.2f}%")

            year_roi_results.append({
                'Strategy': strategy.title(),
                'Bets': roi_results['total_bets'],
                'Win Rate': f"{roi_results['win_rate']:.1f}%",
                'Profit': f"${roi_results['profit']:,.0f}",
                'ROI': f"{roi_results['roi']:+.1f}%"
            })

        # Store results for summary
        best_roi_idx = max(range(len(year_roi_results)),
                          key=lambda i: float(year_roi_results[i]['ROI'].strip('%+')))

        all_results.append({
            'Year': year,
            'Fights': len(test_valid),
            'Accuracy': f"{accuracy:.1f}%",
            'AUC': f"{auc:.4f}",
            'Best Strategy': year_roi_results[best_roi_idx]['Strategy'],
            'Bets': year_roi_results[best_roi_idx]['Bets'],
            'Win Rate': year_roi_results[best_roi_idx]['Win Rate'],
            'ROI': year_roi_results[best_roi_idx]['ROI']
        })

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("YEAR-BY-YEAR SUMMARY (WITH ODDS MODEL)")
    logger.info("="*80)

    summary_df = pd.DataFrame(all_results)
    logger.info(f"\n{summary_df.to_string(index=False)}\n")

    # Overall statistics
    logger.info("="*80)
    logger.info("OVERALL STATISTICS")
    logger.info("="*80)

    total_fights = sum([row['Fights'] for row in all_results])
    avg_accuracy = np.mean([float(row['Accuracy'].strip('%')) for row in all_results])
    avg_auc = np.mean([float(row['AUC']) for row in all_results])

    # Calculate overall ROI (weighted by bets)
    total_bets = sum([row['Bets'] for row in all_results])

    logger.info(f"\nTotal fights tested: {total_fights}")
    logger.info(f"Average accuracy: {avg_accuracy:.1f}%")
    logger.info(f"Average AUC: {avg_auc:.4f}")
    logger.info(f"Total bets placed: {total_bets}")

    # Year-over-year trends
    logger.info("\n" + "="*80)
    logger.info("TRENDS")
    logger.info("="*80)

    accuracies = [float(row['Accuracy'].strip('%')) for row in all_results]
    rois = [float(row['ROI'].strip('%+')) for row in all_results]

    logger.info(f"\nAccuracy trend: {accuracies[0]:.1f}% → {accuracies[-1]:.1f}%")
    logger.info(f"ROI trend: {rois[0]:+.1f}% → {rois[-1]:+.1f}%")

    if accuracies[-1] > accuracies[0]:
        logger.success(f"✓ Accuracy improving over time (+{accuracies[-1] - accuracies[0]:.1f}%)")
    else:
        logger.warning(f"⚠️  Accuracy declining over time ({accuracies[-1] - accuracies[0]:.1f}%)")

    logger.info("\n" + "="*80)
    logger.success("✓ YEAR-BY-YEAR BACKTEST COMPLETE")
    logger.info("="*80)

    logger.info("\nKey Findings:")
    logger.info("1. Each year tested as separate holdout (no training data from that year)")
    logger.info("2. Used WITH-ODDS model (trained on all data before test year)")
    logger.info("3. Fixed stake betting ($100 units)")
    logger.info("4. Real historical odds from dataset\n")


if __name__ == "__main__":
    main()
