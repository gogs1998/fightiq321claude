"""
Prediction Tracking System

Tracks all predictions, actual outcomes, and calculates performance metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from loguru import logger
import json


class PredictionTracker:
    """
    Track predictions and outcomes for monitoring model performance

    Logs all predictions to CSV/JSON and calculates running metrics
    """

    def __init__(self, tracking_dir: str = "data/tracking"):
        """
        Initialize prediction tracker

        Args:
            tracking_dir: Directory to store tracking files
        """
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)

        self.predictions_file = self.tracking_dir / "predictions.csv"
        self.metrics_file = self.tracking_dir / "metrics_history.json"

        # Initialize files if they don't exist
        self._init_files()

    def _init_files(self):
        """Initialize tracking files"""
        if not self.predictions_file.exists():
            # Create empty predictions CSV
            df = pd.DataFrame(columns=[
                'prediction_id',
                'prediction_date',
                'event_name',
                'event_date',
                'fighter1',
                'fighter2',
                'predicted_winner',
                'prediction_confidence',
                'fighter1_odds',
                'fighter2_odds',
                'bet_placed',
                'bet_amount',
                'expected_return',
                'actual_winner',
                'outcome_recorded_date',
                'profit_loss',
                'model_version',
                'notes'
            ])
            df.to_csv(self.predictions_file, index=False)
            logger.info(f"Created tracking file: {self.predictions_file}")

        if not self.metrics_file.exists():
            # Create empty metrics history
            metrics = {
                'created_date': datetime.now().isoformat(),
                'total_predictions': 0,
                'total_outcomes_recorded': 0,
                'monthly_metrics': [],
                'weekly_metrics': []
            }
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Created metrics file: {self.metrics_file}")

    def log_prediction(
        self,
        event_name: str,
        event_date: str,
        fighter1: str,
        fighter2: str,
        predicted_winner: str,
        prediction_confidence: float,
        fighter1_odds: float,
        fighter2_odds: float,
        bet_placed: bool = False,
        bet_amount: float = 0.0,
        model_version: str = "production",
        notes: str = ""
    ) -> str:
        """
        Log a new prediction

        Args:
            event_name: Name of UFC event
            event_date: Date of event (YYYY-MM-DD)
            fighter1: Name of first fighter
            fighter2: Name of second fighter
            predicted_winner: Predicted winner (fighter1 or fighter2)
            prediction_confidence: Confidence score (0-1)
            fighter1_odds: Decimal odds for fighter1
            fighter2_odds: Decimal odds for fighter2
            bet_placed: Whether a bet was placed
            bet_amount: Amount bet (if any)
            model_version: Model version used
            notes: Additional notes

        Returns:
            Prediction ID
        """
        # Generate prediction ID
        prediction_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{fighter1[:4]}_{fighter2[:4]}"

        # Calculate expected return
        if bet_placed:
            winning_odds = fighter1_odds if predicted_winner == fighter1 else fighter2_odds
            expected_return = bet_amount * winning_odds
        else:
            expected_return = 0.0

        # Create prediction record
        prediction = {
            'prediction_id': prediction_id,
            'prediction_date': datetime.now().isoformat(),
            'event_name': event_name,
            'event_date': event_date,
            'fighter1': fighter1,
            'fighter2': fighter2,
            'predicted_winner': predicted_winner,
            'prediction_confidence': prediction_confidence,
            'fighter1_odds': fighter1_odds,
            'fighter2_odds': fighter2_odds,
            'bet_placed': bet_placed,
            'bet_amount': bet_amount,
            'expected_return': expected_return,
            'actual_winner': None,
            'outcome_recorded_date': None,
            'profit_loss': None,
            'model_version': model_version,
            'notes': notes
        }

        # Append to CSV
        df = pd.DataFrame([prediction])
        df.to_csv(self.predictions_file, mode='a', header=False, index=False)

        logger.success(f"✓ Logged prediction {prediction_id}")
        logger.info(f"  {fighter1} vs {fighter2}")
        logger.info(f"  Predicted: {predicted_winner} ({prediction_confidence:.1%} confidence)")

        return prediction_id

    def record_outcome(
        self,
        prediction_id: str,
        actual_winner: str
    ) -> bool:
        """
        Record the actual outcome of a fight

        Args:
            prediction_id: ID of the prediction to update
            actual_winner: Actual winner (fighter name)

        Returns:
            True if successful, False otherwise
        """
        # Load predictions
        df = pd.read_csv(self.predictions_file)

        # Find prediction
        mask = df['prediction_id'] == prediction_id
        if not mask.any():
            logger.error(f"Prediction {prediction_id} not found")
            return False

        # Update outcome
        df.loc[mask, 'actual_winner'] = actual_winner
        df.loc[mask, 'outcome_recorded_date'] = datetime.now().isoformat()

        # Calculate profit/loss for bets
        row = df.loc[mask].iloc[0]
        if row['bet_placed']:
            if row['predicted_winner'] == actual_winner:
                # Win: profit = (bet_amount * odds) - bet_amount
                winning_odds = row['fighter1_odds'] if actual_winner == row['fighter1'] else row['fighter2_odds']
                profit = (row['bet_amount'] * winning_odds) - row['bet_amount']
            else:
                # Loss: lose entire bet
                profit = -row['bet_amount']

            df.loc[mask, 'profit_loss'] = profit
        else:
            df.loc[mask, 'profit_loss'] = 0.0

        # Save updated predictions
        df.to_csv(self.predictions_file, index=False)

        logger.success(f"✓ Recorded outcome for {prediction_id}")
        logger.info(f"  Actual winner: {actual_winner}")
        if row['bet_placed']:
            profit = df.loc[mask, 'profit_loss'].iloc[0]
            logger.info(f"  Profit/Loss: ${profit:.2f}")

        # Update metrics
        self._update_metrics()

        return True

    def bulk_record_outcomes(
        self,
        outcomes: Dict[str, str]
    ) -> int:
        """
        Record multiple outcomes at once

        Args:
            outcomes: Dict mapping prediction_id -> actual_winner

        Returns:
            Number of outcomes recorded
        """
        count = 0
        for prediction_id, actual_winner in outcomes.items():
            if self.record_outcome(prediction_id, actual_winner):
                count += 1

        logger.success(f"✓ Recorded {count}/{len(outcomes)} outcomes")
        return count

    def _update_metrics(self):
        """Update metrics history"""
        df = pd.read_csv(self.predictions_file)

        # Filter to predictions with outcomes
        completed = df[df['actual_winner'].notna()].copy()

        if len(completed) == 0:
            return

        # Calculate overall metrics
        total_predictions = len(df)
        total_outcomes = len(completed)

        # Accuracy
        completed['correct'] = completed['predicted_winner'] == completed['actual_winner']
        accuracy = completed['correct'].mean()

        # Betting metrics
        bets = completed[completed['bet_placed'] == True]
        if len(bets) > 0:
            total_bet = bets['bet_amount'].sum()
            total_profit = bets['profit_loss'].sum()
            roi = (total_profit / total_bet * 100) if total_bet > 0 else 0
            win_rate = bets['correct'].mean()
        else:
            total_bet = 0
            total_profit = 0
            roi = 0
            win_rate = 0

        # Update metrics file
        metrics = {
            'last_updated': datetime.now().isoformat(),
            'total_predictions': int(total_predictions),
            'total_outcomes_recorded': int(total_outcomes),
            'overall_accuracy': float(accuracy),
            'total_bets_placed': int(len(bets)),
            'total_amount_bet': float(total_bet),
            'total_profit_loss': float(total_profit),
            'roi_percentage': float(roi),
            'betting_win_rate': float(win_rate)
        }

        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

    def get_metrics(self, period: str = "all") -> Dict:
        """
        Get performance metrics

        Args:
            period: Time period ("all", "month", "week", "day")

        Returns:
            Dict of metrics
        """
        df = pd.read_csv(self.predictions_file)
        completed = df[df['actual_winner'].notna()].copy()

        if len(completed) == 0:
            logger.warning("No completed predictions yet")
            return {
                'accuracy': 0,
                'total_predictions': 0,
                'roi': 0,
                'profit_loss': 0
            }

        # Filter by period
        if period != "all":
            completed['outcome_date'] = pd.to_datetime(completed['outcome_recorded_date'])
            now = pd.Timestamp.now()

            if period == "month":
                completed = completed[completed['outcome_date'] > (now - pd.Timedelta(days=30))]
            elif period == "week":
                completed = completed[completed['outcome_date'] > (now - pd.Timedelta(days=7))]
            elif period == "day":
                completed = completed[completed['outcome_date'] > (now - pd.Timedelta(days=1))]

        # Calculate metrics
        completed['correct'] = completed['predicted_winner'] == completed['actual_winner']
        accuracy = completed['correct'].mean()

        bets = completed[completed['bet_placed'] == True]
        if len(bets) > 0:
            total_bet = bets['bet_amount'].sum()
            total_profit = bets['profit_loss'].sum()
            roi = (total_profit / total_bet * 100) if total_bet > 0 else 0
            win_rate = bets['correct'].mean()
        else:
            total_bet = 0
            total_profit = 0
            roi = 0
            win_rate = 0

        return {
            'period': period,
            'total_predictions': len(completed),
            'accuracy': accuracy,
            'bets_placed': len(bets),
            'total_bet': total_bet,
            'profit_loss': total_profit,
            'roi_percentage': roi,
            'win_rate': win_rate
        }

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive performance report

        Args:
            output_file: Optional file path to save report

        Returns:
            Report string
        """
        # Get metrics for different periods
        metrics_all = self.get_metrics("all")
        metrics_month = self.get_metrics("month")
        metrics_week = self.get_metrics("week")

        # Build report
        report = []
        report.append("="*80)
        report.append("FIGHTIQ PREDICTION TRACKING REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Overall performance
        report.append("OVERALL PERFORMANCE (All Time)")
        report.append("-"*80)
        report.append(f"Total Predictions: {metrics_all['total_predictions']}")
        report.append(f"Accuracy: {metrics_all['accuracy']:.1%}")
        report.append(f"Bets Placed: {metrics_all['bets_placed']}")
        report.append(f"Total Bet: ${metrics_all['total_bet']:.2f}")
        report.append(f"Profit/Loss: ${metrics_all['profit_loss']:.2f}")
        report.append(f"ROI: {metrics_all['roi_percentage']:.1f}%")
        report.append(f"Win Rate (Bets): {metrics_all['win_rate']:.1%}\n")

        # Monthly performance
        report.append("LAST 30 DAYS")
        report.append("-"*80)
        report.append(f"Predictions: {metrics_month['total_predictions']}")
        report.append(f"Accuracy: {metrics_month['accuracy']:.1%}")
        report.append(f"Profit/Loss: ${metrics_month['profit_loss']:.2f}")
        report.append(f"ROI: {metrics_month['roi_percentage']:.1f}%\n")

        # Weekly performance
        report.append("LAST 7 DAYS")
        report.append("-"*80)
        report.append(f"Predictions: {metrics_week['total_predictions']}")
        report.append(f"Accuracy: {metrics_week['accuracy']:.1%}")
        report.append(f"Profit/Loss: ${metrics_week['profit_loss']:.2f}")
        report.append(f"ROI: {metrics_week['roi_percentage']:.1f}%\n")

        # Recent predictions
        df = pd.read_csv(self.predictions_file)
        recent = df.tail(10)

        report.append("RECENT PREDICTIONS (Last 10)")
        report.append("-"*80)
        for _, row in recent.iterrows():
            status = "✓" if row['predicted_winner'] == row['actual_winner'] else "✗" if pd.notna(row['actual_winner']) else "⏳"
            report.append(f"{status} {row['fighter1']} vs {row['fighter2']}")
            report.append(f"   Predicted: {row['predicted_winner']} ({row['prediction_confidence']:.0%})")
            if pd.notna(row['actual_winner']):
                report.append(f"   Actual: {row['actual_winner']}")
            if row['bet_placed']:
                report.append(f"   Bet: ${row['bet_amount']:.2f} | P/L: ${row['profit_loss']:.2f}")
            report.append("")

        report_text = "\n".join(report)

        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.success(f"✓ Report saved to: {output_file}")

        return report_text

    def detect_drift(self, window_size: int = 20, threshold: float = 0.05) -> Dict:
        """
        Detect model performance drift

        Args:
            window_size: Number of recent predictions to compare
            threshold: Accuracy drop threshold to trigger alert

        Returns:
            Dict with drift analysis
        """
        df = pd.read_csv(self.predictions_file)
        completed = df[df['actual_winner'].notna()].copy()

        if len(completed) < window_size * 2:
            return {
                'drift_detected': False,
                'reason': 'Insufficient data for drift detection'
            }

        # Calculate rolling accuracy
        completed['correct'] = completed['predicted_winner'] == completed['actual_winner']
        completed = completed.sort_values('prediction_date')

        # Recent window vs baseline
        baseline_accuracy = completed.iloc[:-window_size]['correct'].mean()
        recent_accuracy = completed.iloc[-window_size:]['correct'].mean()

        accuracy_drop = baseline_accuracy - recent_accuracy

        drift_detected = accuracy_drop > threshold

        result = {
            'drift_detected': drift_detected,
            'baseline_accuracy': baseline_accuracy,
            'recent_accuracy': recent_accuracy,
            'accuracy_drop': accuracy_drop,
            'threshold': threshold,
            'window_size': window_size
        }

        if drift_detected:
            logger.warning("⚠️  MODEL DRIFT DETECTED!")
            logger.warning(f"  Baseline accuracy: {baseline_accuracy:.1%}")
            logger.warning(f"  Recent accuracy: {recent_accuracy:.1%}")
            logger.warning(f"  Drop: {accuracy_drop:.1%} (threshold: {threshold:.1%})")
            logger.info("\nRecommended actions:")
            logger.info("  1. Review recent predictions for patterns")
            logger.info("  2. Check for distribution shifts in data")
            logger.info("  3. Consider retraining the model")

        return result


def main():
    """Example usage"""
    logger.info("="*80)
    logger.info("PREDICTION TRACKER - DEMO")
    logger.info("="*80 + "\n")

    tracker = PredictionTracker()

    # Example: Log a prediction
    pred_id = tracker.log_prediction(
        event_name="UFC 321: Aspinall vs Gane",
        event_date="2025-10-25",
        fighter1="Tom Aspinall",
        fighter2="Ciryl Gane",
        predicted_winner="Tom Aspinall",
        prediction_confidence=0.78,
        fighter1_odds=1.85,
        fighter2_odds=2.15,
        bet_placed=True,
        bet_amount=100.0,
        model_version="ensemble_production_v1.2",
        notes="High confidence prediction based on striking differential"
    )

    logger.info(f"\n✓ Prediction logged with ID: {pred_id}")

    # Example: Record outcome (do this after the fight)
    # tracker.record_outcome(pred_id, "Tom Aspinall")

    # Generate report
    logger.info("\n" + tracker.generate_report())


if __name__ == "__main__":
    main()
