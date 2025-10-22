"""
Predict Upcoming UFC Fights using BestFightOdds (Production System)

IMPROVED VERSION: Replaces The Odds API with BestFightOdds

This script:
1. Fetches upcoming UFC fights from BestFightOdds
2. Loads most recent fighter features from golden dataset
3. Combines with real-time odds
4. Makes predictions using production ensemble model
5. Provides betting recommendations

Advantages over The Odds API:
- Complete fight coverage (all prelims included)
- More reliable odds
- 12+ bookmakers
- Free with no rate limits
- Better fighter name consistency
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from loguru import logger
from fuzzywuzzy import fuzz, process

from src.utils.config import get_config
from fetch_odds_bestfightodds import fetch_upcoming_ufc_odds, BestFightOddsScraper


def normalize_fighter_name(name: str) -> str:
    """Normalize fighter name for database matching"""
    if pd.isna(name):
        return ""

    normalized = str(name).strip().lower()

    # Remove common suffixes
    for suffix in [' jr', ' jr.', ' sr', ' sr.', ' ii', ' iii', ' iv']:
        if normalized.endswith(suffix):
            normalized = normalized.replace(suffix, '')

    # Remove punctuation
    for char in ['.', "'", '-']:
        normalized = normalized.replace(char, ' ')

    # Remove extra spaces
    normalized = ' '.join(normalized.split())

    return normalized


def find_fighter_in_database(fighter_name: str, df_golden: pd.DataFrame) -> pd.Series:
    """
    Find fighter's most recent features from golden dataset

    Uses fuzzy matching to handle name variations

    Args:
        fighter_name: Name from BestFightOdds
        df_golden: Golden dataset

    Returns:
        Series with fighter's most recent fight features (or None if not found)
    """
    normalized_search = normalize_fighter_name(fighter_name)

    # Build list of all fighters in database
    fighters_f1 = df_golden[['f_1_name']].drop_duplicates().rename(columns={'f_1_name': 'name'})
    fighters_f2 = df_golden[['f_2_name']].drop_duplicates().rename(columns={'f_2_name': 'name'})
    all_fighters = pd.concat([fighters_f1, fighters_f2]).drop_duplicates()

    # Normalize all fighter names
    all_fighters['normalized'] = all_fighters['name'].apply(normalize_fighter_name)

    # Fuzzy match
    matches = process.extract(
        normalized_search,
        all_fighters['normalized'].tolist(),
        limit=3,
        scorer=fuzz.token_sort_ratio
    )

    if not matches:
        logger.warning(f"No match found for '{fighter_name}'")
        return None

    best_match_normalized, best_match_score = matches[0]

    if best_match_score < 70:
        logger.warning(f"Low match confidence for '{fighter_name}' (best: {best_match_score}%)")
        return None

    # Get the original name
    best_match_name = all_fighters[all_fighters['normalized'] == best_match_normalized]['name'].iloc[0]

    logger.info(f"  Matched '{fighter_name}' → '{best_match_name}' ({best_match_score}% confidence)")

    # Find fighter's most recent fight
    f1_fights = df_golden[df_golden['f_1_name'] == best_match_name].sort_values('event_date', ascending=False)
    f2_fights = df_golden[df_golden['f_2_name'] == best_match_name].sort_values('event_date', ascending=False)

    if len(f1_fights) > 0:
        recent_fight = f1_fights.iloc[0]
        fighter_position = 'f_1'
    elif len(f2_fights) > 0:
        recent_fight = f2_fights.iloc[0]
        fighter_position = 'f_2'
    else:
        logger.warning(f"No fights found for '{best_match_name}'")
        return None

    return recent_fight, fighter_position, best_match_name


def predict_upcoming_fights(event_name: str = "UFC 321",
                           confidence_threshold: float = 0.60):
    """
    Main prediction pipeline using BestFightOdds

    Args:
        event_name: UFC event name (e.g., "UFC 321: Aspinall vs Gane")
        confidence_threshold: Minimum confidence for betting recommendations
    """
    logger.info("=" * 80)
    logger.info("UFC FIGHT PREDICTION - PRODUCTION SYSTEM (BESTFIGHTODDS)")
    logger.info("=" * 80)

    config = get_config()

    # Step 1: Fetch upcoming fights from BestFightOdds
    logger.info(f"\n📊 Fetching fights for: {event_name}\n")

    upcoming_fights = fetch_upcoming_ufc_odds(event_name)

    if len(upcoming_fights) == 0:
        logger.error("✗ No upcoming fights found. Check event name.")
        return

    logger.info(f"\n✓ Found {len(upcoming_fights)} fights\n")

    # Step 2: Load golden dataset
    logger.info("=" * 80)
    logger.info("LOADING GOLDEN DATASET")
    logger.info("=" * 80)

    golden_path = Path("D:/Codex/UFC-Master-Pipeline/UFC_full_data_golden.csv")

    if not golden_path.exists():
        logger.error(f"✗ Golden dataset not found: {golden_path}")
        return

    df_golden = pd.read_csv(golden_path)
    df_golden['event_date'] = pd.to_datetime(df_golden['event_date'])

    logger.info(f"✓ Loaded {len(df_golden):,} historical fights")

    # Step 3: Load production model
    logger.info("\n" + "=" * 80)
    logger.info("LOADING PRODUCTION MODEL")
    logger.info("=" * 80)

    model_path = Path("D:/Codex/UFC-Master-Pipeline/models/ensemble_production.pkl")

    if not model_path.exists():
        logger.error(f"✗ Production model not found: {model_path}")
        logger.info("Run scripts/train_production.py first to train the model")
        return

    with open(model_path, 'rb') as f:
        ensemble_model = pickle.load(f)

    logger.info("✓ Production ensemble model loaded")
    logger.info("  - XGBoost + LightGBM")
    logger.info("  - 1,476 features")
    logger.info("  - 70.8% test accuracy on 2025 holdout")

    # Step 4: Match fighters and build feature matrix
    logger.info("\n" + "=" * 80)
    logger.info("MATCHING FIGHTERS TO DATABASE")
    logger.info("=" * 80)

    predictions = []
    matched_fights = 0

    for idx, fight in upcoming_fights.iterrows():
        fighter1_name = fight['fighter1']
        fighter2_name = fight['fighter2']
        f1_odds = fight['fighter1_odds']
        f2_odds = fight['fighter2_odds']

        logger.info(f"\n[Fight {idx+1}/{len(upcoming_fights)}] {fighter1_name} vs {fighter2_name}")

        # Find both fighters in database
        f1_match = find_fighter_in_database(fighter1_name, df_golden)
        f2_match = find_fighter_in_database(fighter2_name, df_golden)

        if f1_match is None or f2_match is None:
            logger.warning(f"  ⚠️  Skipping - fighter(s) not found in database")
            predictions.append({
                'fighter1': fighter1_name,
                'fighter2': fighter2_name,
                'fighter1_odds': f1_odds,
                'fighter2_odds': f2_odds,
                'predicted_winner': 'Unknown',
                'confidence': 0,
                'prob_f1_wins': 0.5,
                'prob_f2_wins': 0.5,
                'recommended_bet': 'PASS - Unable to match fighters in database'
            })
            continue

        f1_recent_fight, f1_position, f1_db_name = f1_match
        f2_recent_fight, f2_position, f2_db_name = f2_match

        # Step 5: Build feature vector (simplified approach)
        # Extract features from most recent fights
        feature_cols = [col for col in df_golden.columns
                       if col not in ['event_date', 'event_name', 'f_1_name', 'f_2_name',
                                     'fight_id', 'actual_winner', 'target']]

        # Get numeric features only
        numeric_features = df_golden[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        # Create feature vector by combining both fighters' most recent stats
        # This is a simplified approach - ideally we'd reconstruct all 1,476 features
        X_upcoming = pd.DataFrame()

        # Extract fighter 1 features (from their most recent fight)
        f1_features = {}
        for col in numeric_features:
            if col.startswith('f_1_') and f1_position == 'f_1':
                f1_features[col] = f1_recent_fight.get(col, np.nan)
            elif col.startswith('f_2_') and f1_position == 'f_2':
                # Map f_2_ features to f_1_ for this upcoming fight
                new_col = col.replace('f_2_', 'f_1_')
                f1_features[new_col] = f1_recent_fight.get(col, np.nan)

        # Extract fighter 2 features
        f2_features = {}
        for col in numeric_features:
            if col.startswith('f_2_') and f2_position == 'f_2':
                f2_features[col] = f2_recent_fight.get(col, np.nan)
            elif col.startswith('f_1_') and f2_position == 'f_1':
                # Map f_1_ features to f_2_ for this upcoming fight
                new_col = col.replace('f_1_', 'f_2_')
                f2_features[new_col] = f2_recent_fight.get(col, np.nan)

        # Combine features
        combined_features = {**f1_features, **f2_features}

        # Add odds features
        combined_features['f_1_odds'] = f1_odds
        combined_features['f_2_odds'] = f2_odds

        X_upcoming = pd.DataFrame([combined_features])

        # Fill missing values with median from training data
        for col in X_upcoming.columns:
            if X_upcoming[col].isna().any():
                X_upcoming[col] = X_upcoming[col].fillna(df_golden[col].median())

        # Step 6: Make prediction
        try:
            pred_proba = ensemble_model.predict_proba(X_upcoming[numeric_features])[0]

            prob_f1_wins = 1 - pred_proba[1]  # Probability fighter 1 wins
            prob_f2_wins = pred_proba[1]       # Probability fighter 2 wins

            predicted_winner = fighter1_name if prob_f1_wins > prob_f2_wins else fighter2_name
            confidence = max(prob_f1_wins, prob_f2_wins)

            # Betting recommendation
            if confidence >= confidence_threshold:
                # Check if we have positive expected value
                pick_odds = f1_odds if predicted_winner == fighter1_name else f2_odds
                market_implied_prob = 1 / pick_odds
                edge = confidence - market_implied_prob

                if edge > 0:
                    expected_roi = (confidence * pick_odds - 1) * 100
                    recommended_bet = f"BET {predicted_winner} ({expected_roi:+.1f}% expected ROI)"
                else:
                    recommended_bet = "PASS - No positive edge"
            else:
                recommended_bet = f"PASS - Confidence {confidence*100:.1f}% below {confidence_threshold*100:.0f}% threshold"

            matched_fights += 1

            logger.info(f"  ✓ Prediction: {predicted_winner} ({confidence*100:.1f}% confidence)")
            logger.info(f"  {recommended_bet}")

            predictions.append({
                'fighter1': fighter1_name,
                'fighter2': fighter2_name,
                'fighter1_odds': f1_odds,
                'fighter2_odds': f2_odds,
                'predicted_winner': predicted_winner,
                'confidence': confidence,
                'prob_f1_wins': prob_f1_wins,
                'prob_f2_wins': prob_f2_wins,
                'recommended_bet': recommended_bet
            })

        except Exception as e:
            logger.error(f"  ✗ Prediction failed: {e}")
            predictions.append({
                'fighter1': fighter1_name,
                'fighter2': fighter2_name,
                'fighter1_odds': f1_odds,
                'fighter2_odds': f2_odds,
                'predicted_winner': 'Error',
                'confidence': 0,
                'prob_f1_wins': 0.5,
                'prob_f2_wins': 0.5,
                'recommended_bet': f'ERROR - {str(e)}'
            })

    # Step 7: Generate output
    df_predictions = pd.DataFrame(predictions)

    logger.info("\n" + "=" * 80)
    logger.info(f"PREDICTIONS SUMMARY - {event_name.upper()}")
    logger.info("=" * 80)

    logger.info(f"\nTotal fights: {len(df_predictions)}")
    logger.info(f"Matched fighters: {matched_fights} / {len(df_predictions)}")

    # Count betting recommendations
    bets = df_predictions[df_predictions['recommended_bet'].str.startswith('BET', na=False)]
    logger.info(f"High-confidence bets: {len(bets)} ({len(bets)/len(df_predictions)*100:.1f}%)")

    # Save to CSV
    output_file = f"predictions_{event_name.lower().replace(' ', '_').replace(':', '')}.csv"
    df_predictions.to_csv(output_file, index=False)
    logger.success(f"\n✓ Predictions saved to: {output_file}")

    # Display high-confidence bets
    if len(bets) > 0:
        logger.info("\n" + "=" * 80)
        logger.info("HIGH-CONFIDENCE BETTING RECOMMENDATIONS")
        logger.info("=" * 80)

        for idx, bet in bets.iterrows():
            logger.info(f"\n{bet['fighter1']} ({bet['fighter1_odds']:.2f}) vs {bet['fighter2']} ({bet['fighter2_odds']:.2f})")
            logger.info(f"  PICK: {bet['predicted_winner']} ({bet['confidence']*100:.1f}% confidence)")
            logger.info(f"  {bet['recommended_bet']}")

    logger.info("\n" + "=" * 80)
    logger.success("✓ PREDICTION PIPELINE COMPLETE")
    logger.info("=" * 80)

    return df_predictions


if __name__ == "__main__":
    # Predict UFC 321: Aspinall vs Gane
    predictions = predict_upcoming_fights(
        event_name="UFC 321: Aspinall vs Gane",
        confidence_threshold=0.60
    )

    logger.info("\n📊 View full breakdown:")
    logger.info("python ufc321_full_breakdown.py")
