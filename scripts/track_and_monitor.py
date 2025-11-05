"""
Prediction Tracking and Monitoring Script

Generates performance reports and checks for model drift
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.tracking.prediction_tracker import PredictionTracker
from loguru import logger
import argparse


def generate_report(tracker: PredictionTracker, save_to_file: bool = True):
    """Generate and display performance report"""
    logger.info("="*80)
    logger.info("GENERATING PERFORMANCE REPORT")
    logger.info("="*80 + "\n")

    output_file = None
    if save_to_file:
        output_file = Path("reports") / f"performance_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt"
        output_file.parent.mkdir(exist_ok=True)

    report = tracker.generate_report(output_file=str(output_file) if output_file else None)
    print(report)


def check_drift(tracker: PredictionTracker, window_size: int = 20):
    """Check for model performance drift"""
    logger.info("="*80)
    logger.info("CHECKING FOR MODEL DRIFT")
    logger.info("="*80 + "\n")

    drift_result = tracker.detect_drift(window_size=window_size)

    if drift_result['drift_detected']:
        logger.warning("⚠️  ALERT: Model drift detected!")
        logger.info(f"\nBaseline accuracy: {drift_result['baseline_accuracy']:.1%}")
        logger.info(f"Recent accuracy: {drift_result['recent_accuracy']:.1%}")
        logger.info(f"Accuracy drop: {drift_result['accuracy_drop']:.1%}")
        logger.info(f"\nAction required: Consider retraining the model")
    else:
        logger.success("✓ No significant drift detected")
        logger.info(f"Recent accuracy: {drift_result.get('recent_accuracy', 0):.1%}")


def record_outcomes_interactive(tracker: PredictionTracker):
    """Interactively record fight outcomes"""
    import pandas as pd

    logger.info("="*80)
    logger.info("RECORD FIGHT OUTCOMES")
    logger.info("="*80 + "\n")

    # Load predictions without outcomes
    df = pd.read_csv(tracker.predictions_file)
    pending = df[df['actual_winner'].isna()].copy()

    if len(pending) == 0:
        logger.info("✓ No pending outcomes to record")
        return

    logger.info(f"Found {len(pending)} predictions without outcomes\n")

    # Group by event for easier recording
    for event_name, group in pending.groupby('event_name'):
        logger.info(f"\nEvent: {event_name}")
        logger.info("-"*60)

        for idx, row in group.iterrows():
            logger.info(f"\n{row['fighter1']} vs {row['fighter2']}")
            logger.info(f"Predicted: {row['predicted_winner']} ({row['prediction_confidence']:.0%})")

            # Ask for outcome
            while True:
                response = input(f"Who won? (1={row['fighter1']}, 2={row['fighter2']}, s=skip): ").strip().lower()

                if response == 's':
                    logger.info("Skipped")
                    break
                elif response == '1':
                    tracker.record_outcome(row['prediction_id'], row['fighter1'])
                    break
                elif response == '2':
                    tracker.record_outcome(row['prediction_id'], row['fighter2'])
                    break
                else:
                    logger.warning("Invalid input. Try again.")

    logger.success("\n✓ Outcome recording complete")


def record_outcomes_from_file(tracker: PredictionTracker, outcomes_file: str):
    """
    Record outcomes from a CSV file

    CSV format: prediction_id, actual_winner
    """
    import pandas as pd

    logger.info(f"Loading outcomes from: {outcomes_file}")

    try:
        outcomes_df = pd.read_csv(outcomes_file)

        if 'prediction_id' not in outcomes_df.columns or 'actual_winner' not in outcomes_df.columns:
            logger.error("CSV must have columns: prediction_id, actual_winner")
            return

        outcomes = dict(zip(outcomes_df['prediction_id'], outcomes_df['actual_winner']))
        count = tracker.bulk_record_outcomes(outcomes)

        logger.success(f"✓ Recorded {count} outcomes from file")

    except Exception as e:
        logger.error(f"Failed to load outcomes file: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Track and monitor FightIQ predictions")

    parser.add_argument(
        '--action',
        choices=['report', 'drift', 'record', 'record-file'],
        default='report',
        help="Action to perform"
    )

    parser.add_argument(
        '--tracking-dir',
        default='data/tracking',
        help="Directory containing tracking data"
    )

    parser.add_argument(
        '--window-size',
        type=int,
        default=20,
        help="Window size for drift detection"
    )

    parser.add_argument(
        '--outcomes-file',
        help="CSV file with outcomes (for record-file action)"
    )

    args = parser.parse_args()

    # Initialize tracker
    tracker = PredictionTracker(tracking_dir=args.tracking_dir)

    # Perform action
    if args.action == 'report':
        generate_report(tracker)

    elif args.action == 'drift':
        check_drift(tracker, window_size=args.window_size)

    elif args.action == 'record':
        record_outcomes_interactive(tracker)

    elif args.action == 'record-file':
        if not args.outcomes_file:
            logger.error("--outcomes-file required for record-file action")
            return
        record_outcomes_from_file(tracker, args.outcomes_file)


if __name__ == "__main__":
    import pandas as pd
    main()
