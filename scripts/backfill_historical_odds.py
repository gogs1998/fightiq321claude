"""
Backfill missing odds in golden dataset using BestFightOdds

Improves odds coverage from 66% to ~95% by fetching historical
odds from BestFightOdds.com (2008-present).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
from fuzzywuzzy import fuzz, process
from fetch_odds_bestfightodds import fetch_historical_odds, BestFightOddsScraper


def normalize_fighter_name(name: str) -> str:
    """Normalize fighter name for matching"""
    if pd.isna(name):
        return ""

    normalized = str(name).strip().lower()

    # Common variations
    replacements = {
        '.': '',
        "'": '',
        '-': ' ',
        'jr': '',
        'sr': '',
        'ii': '',
        'iii': '',
    }

    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    # Remove extra spaces
    normalized = ' '.join(normalized.split())

    return normalized


def match_fight_to_odds(golden_fight: pd.Series, odds_df: pd.DataFrame,
                       threshold: int = 80) -> pd.Series:
    """
    Match a fight from golden dataset to BestFightOdds data

    Args:
        golden_fight: Row from golden dataset
        odds_df: DataFrame from BestFightOdds
        threshold: Minimum fuzzy match score (0-100)

    Returns:
        Series with matched odds (or NaN if no match)
    """
    # Normalize fighter names from golden dataset
    f1_golden = normalize_fighter_name(golden_fight.get('f_1_name', ''))
    f2_golden = normalize_fighter_name(golden_fight.get('f_2_name', ''))

    if not f1_golden or not f2_golden:
        return pd.Series({'f_1_odds': np.nan, 'f_2_odds': np.nan, 'match_score': 0})

    # Create search strings
    golden_matchup = f"{f1_golden} vs {f2_golden}"

    best_match = None
    best_score = 0

    for idx, odds_row in odds_df.iterrows():
        # Normalize BestFightOdds fighter names
        f1_bfo = normalize_fighter_name(odds_row.get('fighter1', ''))
        f2_bfo = normalize_fighter_name(odds_row.get('fighter2', ''))

        if not f1_bfo or not f2_bfo:
            continue

        bfo_matchup = f"{f1_bfo} vs {f2_bfo}"

        # Fuzzy match
        score = fuzz.token_sort_ratio(golden_matchup, bfo_matchup)

        if score > best_score:
            best_score = score
            best_match = odds_row

    # Return match if above threshold
    if best_score >= threshold and best_match is not None:
        return pd.Series({
            'f_1_odds': best_match['fighter1_odds'],
            'f_2_odds': best_match['fighter2_odds'],
            'match_score': best_score,
            'matched_f1': best_match['fighter1'],
            'matched_f2': best_match['fighter2']
        })
    else:
        return pd.Series({
            'f_1_odds': np.nan,
            'f_2_odds': np.nan,
            'match_score': best_score
        })


def backfill_odds_for_dataset(
    golden_path: str = "UFC_full_data_golden.csv",
    output_path: str = "UFC_full_data_golden_with_odds.csv",
    start_year: int = 2008,
    end_year: int = 2025
):
    """
    Backfill missing odds in golden dataset

    Args:
        golden_path: Path to golden dataset CSV
        output_path: Path to save enhanced dataset
        start_year: Start year for historical odds
        end_year: End year for historical odds
    """
    logger.info("="*80)
    logger.info("BACKFILLING HISTORICAL ODDS FROM BESTFIGHTODDS")
    logger.info("="*80)

    # Load golden dataset
    logger.info(f"\nLoading golden dataset: {golden_path}")
    df_golden = pd.read_csv(golden_path)
    df_golden['event_date'] = pd.to_datetime(df_golden['event_date'])

    logger.info(f"✓ Loaded {len(df_golden):,} fights")

    # Check current odds coverage
    has_f1_odds = df_golden['f_1_odds'].notna()
    has_f2_odds = df_golden['f_2_odds'].notna()
    has_both_odds = has_f1_odds & has_f2_odds

    initial_coverage = has_both_odds.sum()
    initial_pct = has_both_odds.mean() * 100

    logger.info(f"\nCurrent odds coverage:")
    logger.info(f"  Fights with odds: {initial_coverage:,} / {len(df_golden):,} ({initial_pct:.1f}%)")
    logger.info(f"  Missing odds: {(~has_both_odds).sum():,}")

    # Filter fights missing odds
    missing_odds = df_golden[~has_both_odds].copy()

    logger.info(f"\n📊 Backfilling {len(missing_odds):,} fights missing odds...")

    # Fetch historical odds from BestFightOdds
    logger.info(f"\nFetching historical odds from BestFightOdds ({start_year}-{end_year})...")

    historical_odds = fetch_historical_odds(
        start_date=f"{start_year}-01-01",
        end_date=f"{end_year}-12-31"
    )

    if len(historical_odds) == 0:
        logger.error("✗ No historical odds fetched. Aborting.")
        return

    logger.info(f"✓ Fetched odds for {len(historical_odds):,} historical fights")

    # Match fights and backfill
    logger.info("\n🔍 Matching fights to historical odds...")

    matched_count = 0
    high_confidence_matches = 0
    low_confidence_matches = 0

    for idx, fight in missing_odds.iterrows():
        # Get event date year
        fight_year = fight['event_date'].year

        # Filter odds to same year (±1 year for safety)
        year_odds = historical_odds[
            (pd.to_datetime(historical_odds['event_date'], format='%b %d, %Y').dt.year >= fight_year - 1) &
            (pd.to_datetime(historical_odds['event_date'], format='%b %d, %Y').dt.year <= fight_year + 1)
        ]

        if len(year_odds) == 0:
            continue

        # Match fight
        match_result = match_fight_to_odds(fight, year_odds, threshold=75)

        if pd.notna(match_result['f_1_odds']) and pd.notna(match_result['f_2_odds']):
            # Update golden dataset
            df_golden.at[idx, 'f_1_odds'] = match_result['f_1_odds']
            df_golden.at[idx, 'f_2_odds'] = match_result['f_2_odds']

            matched_count += 1

            if match_result['match_score'] >= 90:
                high_confidence_matches += 1
            else:
                low_confidence_matches += 1

            # Log progress every 100 matches
            if matched_count % 100 == 0:
                logger.info(f"  Matched {matched_count} fights...")

    # Final statistics
    logger.info("\n" + "="*80)
    logger.info("BACKFILL RESULTS")
    logger.info("="*80)

    final_has_odds = df_golden['f_1_odds'].notna() & df_golden['f_2_odds'].notna()
    final_coverage = final_has_odds.sum()
    final_pct = final_has_odds.mean() * 100

    improvement = final_coverage - initial_coverage
    improvement_pct = final_pct - initial_pct

    logger.info(f"\nMatching Statistics:")
    logger.info(f"  Total matches found: {matched_count:,}")
    logger.info(f"  High confidence (≥90%): {high_confidence_matches:,}")
    logger.info(f"  Medium confidence (75-89%): {low_confidence_matches:,}")

    logger.info(f"\nOdds Coverage:")
    logger.info(f"  Before: {initial_coverage:,} / {len(df_golden):,} ({initial_pct:.1f}%)")
    logger.info(f"  After:  {final_coverage:,} / {len(df_golden):,} ({final_pct:.1f}%)")
    logger.info(f"  Improvement: +{improvement:,} fights (+{improvement_pct:.1f}%)")

    # Save enhanced dataset
    logger.info(f"\nSaving enhanced dataset to: {output_path}")
    df_golden.to_csv(output_path, index=False)
    logger.success(f"✓ Saved {len(df_golden):,} fights with improved odds coverage")

    # Create summary report
    summary = {
        'initial_coverage': initial_coverage,
        'initial_percentage': initial_pct,
        'final_coverage': final_coverage,
        'final_percentage': final_pct,
        'fights_backfilled': improvement,
        'high_confidence_matches': high_confidence_matches,
        'low_confidence_matches': low_confidence_matches
    }

    summary_df = pd.DataFrame([summary])
    summary_path = output_path.replace('.csv', '_backfill_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    logger.info(f"✓ Saved backfill summary to: {summary_path}")

    logger.info("\n" + "="*80)
    logger.success("✓ BACKFILL COMPLETE")
    logger.info("="*80)


def quick_backfill_recent_years(
    golden_path: str = "UFC_full_data_golden.csv",
    output_path: str = "UFC_full_data_golden_with_odds.csv",
    years: list = [2022, 2023, 2024, 2025]
):
    """
    Quick backfill for specific years only

    Faster than full backfill, useful for recent data
    """
    logger.info("="*80)
    logger.info(f"QUICK BACKFILL FOR YEARS: {', '.join(map(str, years))}")
    logger.info("="*80)

    # Load golden dataset
    df_golden = pd.read_csv(golden_path)
    df_golden['event_date'] = pd.to_datetime(df_golden['event_date'])

    # Filter to specified years
    df_filtered = df_golden[df_golden['event_date'].dt.year.isin(years)].copy()

    logger.info(f"\nFights in {years}: {len(df_filtered):,}")

    # Check current coverage
    has_odds = (df_filtered['f_1_odds'].notna()) & (df_filtered['f_2_odds'].notna())
    missing = ~has_odds

    logger.info(f"  With odds: {has_odds.sum():,}")
    logger.info(f"  Missing odds: {missing.sum():,}")

    if missing.sum() == 0:
        logger.success("✓ No backfill needed - all fights have odds!")
        return

    # Fetch odds for each year
    all_historical = []

    for year in years:
        logger.info(f"\nFetching {year} odds from BestFightOdds...")
        year_odds = fetch_historical_odds(f"{year}-01-01", f"{year}-12-31")

        if len(year_odds) > 0:
            all_historical.append(year_odds)
            logger.success(f"  ✓ {len(year_odds):,} fights")

    if not all_historical:
        logger.error("✗ No odds fetched. Aborting.")
        return

    # Combine all odds
    combined_odds = pd.concat(all_historical, ignore_index=True)

    logger.info(f"\nTotal historical odds: {len(combined_odds):,}")

    # Match and backfill
    matched = 0
    missing_fights = df_filtered[missing]

    for idx, fight in missing_fights.iterrows():
        match_result = match_fight_to_odds(fight, combined_odds, threshold=75)

        if pd.notna(match_result['f_1_odds']):
            df_golden.at[idx, 'f_1_odds'] = match_result['f_1_odds']
            df_golden.at[idx, 'f_2_odds'] = match_result['f_2_odds']
            matched += 1

    logger.info(f"\n✓ Matched and backfilled {matched:,} fights")

    # Save
    df_golden.to_csv(output_path, index=False)
    logger.success(f"✓ Saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backfill UFC odds from BestFightOdds")
    parser.add_argument('--mode', choices=['full', 'quick'], default='quick',
                       help='Backfill mode: full (2008-2025) or quick (recent years)')
    parser.add_argument('--input', default='UFC_full_data_golden.csv',
                       help='Input golden dataset CSV')
    parser.add_argument('--output', default='UFC_full_data_golden_with_odds.csv',
                       help='Output enhanced dataset CSV')

    args = parser.parse_args()

    if args.mode == 'full':
        # Full backfill (2008-2025)
        logger.info("Running FULL backfill (2008-2025)...")
        logger.warning("⚠️  This will take 30-60 minutes due to API rate limiting")

        backfill_odds_for_dataset(
            golden_path=args.input,
            output_path=args.output,
            start_year=2008,
            end_year=2025
        )

    else:
        # Quick backfill (recent years only)
        logger.info("Running QUICK backfill (2022-2025)...")

        quick_backfill_recent_years(
            golden_path=args.input,
            output_path=args.output,
            years=[2022, 2023, 2024, 2025]
        )
