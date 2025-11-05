"""
Predict UFC Fights with Automatic Tracking

This wraps the prediction pipeline to automatically log all predictions
to the tracking system for performance monitoring.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from src.tracking.prediction_tracker import PredictionTracker
from fetch_odds_bestfightodds import BestFightOddsScraper


def predict_and_track(
    event_name: str,
    event_date: str,
    confidence_threshold: float = 0.60,
    bet_unit: float = 100.0,
    enable_betting: bool = False,
    tracking_dir: str = "data/tracking"
):
    """
    Make predictions for an event and log them to tracking system

    Args:
        event_name: UFC event name (e.g., "UFC 321")
        event_date: Event date (YYYY-MM-DD)
        confidence_threshold: Minimum confidence to recommend bet
        bet_unit: Standard betting unit ($)
        enable_betting: Whether to actually log bets (vs just predictions)
        tracking_dir: Directory for tracking data

    Returns:
        DataFrame of predictions
    """
    logger.info("="*80)
    logger.info(f"PREDICTING: {event_name}")
    logger.info(f"Date: {event_date}")
    logger.info("="*80 + "\n")

    # Initialize tracker
    tracker = PredictionTracker(tracking_dir=tracking_dir)

    # Step 1: Fetch odds from BestFightOdds
    logger.info("Fetching odds from BestFightOdds...")
    scraper = BestFightOddsScraper(use_selenium=True, headless=True)
    odds_df = scraper.get_consensus_odds(event_name)

    if len(odds_df) == 0:
        logger.error("No odds found. Cannot make predictions.")
        return pd.DataFrame()

    logger.success(f"✓ Found odds for {len(odds_df)} fights\n")

    # Step 2: Load golden dataset
    logger.info("Loading fighter database...")
    config_path = Path(__file__).parents[1] / "config" / "config.yaml"

    from src.utils.config import get_config
    config = get_config(str(config_path))

    golden_dataset_path = Path(config.paths.golden_dataset)
    df_golden = pd.read_csv(golden_dataset_path)

    logger.success(f"✓ Loaded {len(df_golden)} historical fights\n")

    # Step 3: Load production model
    logger.info("Loading ensemble model...")
    model_path = Path(config.paths.models_dir) / "ensemble_production.pkl"

    import pickle
    with open(model_path, 'rb') as f:
        ensemble_model = pickle.load(f)

    logger.success(f"✓ Model loaded\n")

    # Step 4: Make predictions and log them
    predictions = []

    logger.info(f"Making predictions for {len(odds_df)} fights...")
    logger.info("-"*80 + "\n")

    for idx, row in odds_df.iterrows():
        fighter1 = row['fighter1']
        fighter2 = row['fighter2']
        f1_odds = row['fighter1_odds']
        f2_odds = row['fighter2_odds']

        logger.info(f"Fight {idx+1}: {fighter1} vs {fighter2}")

        # Get fighter features (simplified - would use full feature engineering in production)
        from predict_upcoming_with_bestfightodds import find_fighter_in_database

        f1_match = find_fighter_in_database(fighter1, df_golden)
        f2_match = find_fighter_in_database(fighter2, df_golden)

        if f1_match is None or f2_match is None:
            logger.warning("  ⚠️  Fighter(s) not found in database - skipping\n")
            continue

        # Extract features and make prediction
        # (This is simplified - full version would reconstruct all 1,476 features)
        try:
            # Build feature vector (simplified for demo)
            f1_recent, f1_pos, f1_name = f1_match
            f2_recent, f2_pos, f2_name = f2_match

            # For demo purposes, use a simple feature subset
            # In production, use full feature engineering
            feature_cols = [col for col in df_golden.columns
                           if col not in ['event_date', 'event_name', 'f_1_name', 'f_2_name',
                                         'fight_id', 'actual_winner', 'target']]

            numeric_features = df_golden[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

            # Build features (simplified)
            X = pd.DataFrame([{col: np.nan for col in numeric_features}])

            # Add odds
            if 'f_1_odds' in X.columns:
                X['f_1_odds'] = f1_odds
            if 'f_2_odds' in X.columns:
                X['f_2_odds'] = f2_odds

            # Fill missing with median
            for col in X.columns:
                if X[col].isna().any() and col in df_golden.columns:
                    X[col] = df_golden[col].median()

            # Make prediction
            pred_proba = ensemble_model.predict_proba(X)[0]
            prob_f1_wins = 1 - pred_proba[1]
            prob_f2_wins = pred_proba[1]

            predicted_winner = fighter1 if prob_f1_wins > prob_f2_wins else fighter2
            confidence = max(prob_f1_wins, prob_f2_wins)

            # Determine if we should bet
            pick_odds = f1_odds if predicted_winner == fighter1 else f2_odds
            market_implied_prob = 1 / pick_odds
            edge = confidence - market_implied_prob

            should_bet = (confidence >= confidence_threshold and edge > 0 and enable_betting)

            if should_bet:
                expected_roi = (confidence * pick_odds - 1) * 100
                bet_amount = bet_unit
                recommendation = f"BET ${bet_amount:.0f} on {predicted_winner} (Expected ROI: +{expected_roi:.1f}%)"
            else:
                bet_amount = 0
                if confidence < confidence_threshold:
                    recommendation = f"PASS - Confidence {confidence*100:.1f}% below threshold"
                else:
                    recommendation = f"PASS - No positive edge"

            logger.info(f"  Predicted: {predicted_winner} ({confidence*100:.1f}% confidence)")
            logger.info(f"  {recommendation}\n")

            # Log to tracking system
            prediction_id = tracker.log_prediction(
                event_name=event_name,
                event_date=event_date,
                fighter1=fighter1,
                fighter2=fighter2,
                predicted_winner=predicted_winner,
                prediction_confidence=confidence,
                fighter1_odds=f1_odds,
                fighter2_odds=f2_odds,
                bet_placed=should_bet,
                bet_amount=bet_amount,
                model_version="ensemble_production_v1.0",
                notes=recommendation
            )

            predictions.append({
                'prediction_id': prediction_id,
                'fighter1': fighter1,
                'fighter2': fighter2,
                'predicted_winner': predicted_winner,
                'confidence': confidence,
                'fighter1_odds': f1_odds,
                'fighter2_odds': f2_odds,
                'bet_placed': should_bet,
                'bet_amount': bet_amount,
                'recommendation': recommendation
            })

        except Exception as e:
            logger.error(f"  ✗ Prediction failed: {e}\n")
            continue

    # Summary
    logger.info("="*80)
    logger.info("PREDICTION SUMMARY")
    logger.info("="*80)

    if predictions:
        pred_df = pd.DataFrame(predictions)
        logger.info(f"Total predictions: {len(predictions)}")
        logger.info(f"Bets recommended: {pred_df['bet_placed'].sum()}")
        logger.info(f"Total staked: ${pred_df['bet_amount'].sum():.2f}")

        logger.info("\n✓ All predictions logged to tracking system")
        logger.info(f"Tracking directory: {tracking_dir}")
        logger.info("\nTo view performance later:")
        logger.info("  python scripts/track_and_monitor.py --action report")

        return pred_df
    else:
        logger.warning("No predictions were made")
        return pd.DataFrame()


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Predict UFC fights with tracking")

    parser.add_argument(
        '--event',
        default='UFC 321',
        help='UFC event name'
    )

    parser.add_argument(
        '--date',
        default=datetime.now().strftime('%Y-%m-%d'),
        help='Event date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.60,
        help='Minimum confidence for betting recommendations'
    )

    parser.add_argument(
        '--bet-unit',
        type=float,
        default=100.0,
        help='Standard betting unit ($)'
    )

    parser.add_argument(
        '--enable-betting',
        action='store_true',
        help='Actually log bets (not just predictions)'
    )

    args = parser.parse_args()

    # Make predictions with tracking
    predictions = predict_and_track(
        event_name=args.event,
        event_date=args.date,
        confidence_threshold=args.confidence_threshold,
        bet_unit=args.bet_unit,
        enable_betting=args.enable_betting
    )

    if len(predictions) > 0:
        # Save to CSV
        output_file = Path('predictions') / f"{args.event.replace(' ', '_')}_{args.date}.csv"
        output_file.parent.mkdir(exist_ok=True)
        predictions.to_csv(output_file, index=False)
        logger.success(f"\n✓ Predictions saved to: {output_file}")


if __name__ == "__main__":
    main()
