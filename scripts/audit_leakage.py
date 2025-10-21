"""
Comprehensive Pre-Fight Leakage Audit

Checks EVERY feature to ensure nothing from the current fight
is leaking into predictions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import numpy as np
import re
from loguru import logger
from src.utils.config import get_config


def audit_feature_for_leakage(col_name, sample_values):
    """
    Audit a single feature for potential leakage

    Returns:
        (is_leak, reason)
    """

    # DEFINITE LEAKS - Current fight data
    definite_leak_patterns = [
        (r'.*_r\d+_.*', 'Round-by-round stat (only known after fight)'),
        (r'^r\d+_duration', 'Round duration (only known after fight)'),
        (r'^f_\d+_sig_strikes_(att|succ)', 'Current fight sig strikes'),
        (r'^f_\d+_total_strikes_(att|succ)', 'Current fight total strikes'),
        (r'^f_\d+_takedown_(att|succ)', 'Current fight takedowns'),
        (r'^f_\d+_submission_(att|succ)', 'Current fight submissions'),
        (r'^f_\d+_knockdowns', 'Current fight knockdowns'),
        (r'^f_\d+_ground_strikes', 'Current fight ground strikes'),
        (r'^f_\d+_clinch_strikes', 'Current fight clinch strikes'),
        (r'^winner$', 'Fight winner (result)'),
        (r'^result$', 'Fight result'),
        (r'^result_details', 'Fight result details'),
        (r'^finish_round', 'Finish round'),
        (r'^finish_time', 'Finish time'),
        (r'^time_format', 'Fight time format (sometimes added after)'),
    ]

    for pattern, reason in definite_leak_patterns:
        if re.match(pattern, col_name, re.IGNORECASE):
            return True, f"LEAK: {reason}"

    # SUSPICIOUS PATTERNS - Need manual review
    suspicious_patterns = [
        (r'.*current.*', 'Contains "current" - may be current fight data'),
        (r'.*this_fight.*', 'Contains "this_fight" - likely current fight'),
        (r'.*fight_\d+_.*', 'Indexed fight stat - verify it\'s historical'),
    ]

    for pattern, reason in suspicious_patterns:
        if re.match(pattern, col_name, re.IGNORECASE):
            return None, f"SUSPICIOUS: {reason} - needs review"

    # SAFE PATTERNS - Pre-fight data
    safe_patterns = [
        (r'^f_\d+_.*_(avg|total|pct|rate|share|acc)_\d+$', 'Rolling average (historical)'),
        (r'^f_\d+_fighter_.*', 'Fighter career stat'),
        (r'^f_\d+_.*_(career|life|long).*', 'Career/lifetime stat'),
        (r'.*_odds$', 'Betting odds (available pre-fight)'),
        (r'^diff_.*', 'Differential (calculated from historical stats)'),
        (r'^matchup_.*', 'Matchup feature (calculated from historical stats)'),
        (r'^momentum_.*', 'Momentum feature (from recent historical fights)'),
        (r'.*_(age|height|weight|reach)', 'Physical attributes'),
        (r'.*ranking.*', 'Rankings (known pre-fight)'),
    ]

    for pattern, reason in safe_patterns:
        if re.match(pattern, col_name, re.IGNORECASE):
            return False, f"SAFE: {reason}"

    # Unknown - needs inspection
    return None, "UNKNOWN: Manual inspection needed"


def main():
    logger.info("\n" + "="*80)
    logger.info("PRE-FIGHT LEAKAGE AUDIT")
    logger.info("="*80)

    config = get_config()

    # Load the dataset
    logger.info("\n" + "="*80)
    logger.info("LOADING DATASET")
    logger.info("="*80)

    df = pd.read_csv(config.paths.golden_dataset)
    logger.info(f"✓ Loaded {len(df):,} fights, {len(df.columns)} columns\n")

    # Get features used in with-odds model
    logger.info("="*80)
    logger.info("LOADING WITH-ODDS MODEL FEATURES")
    logger.info("="*80)

    # Simulate the feature loading from train_with_odds.py
    leaking_patterns = [
        r'.*_r\d+_.*',
        r'^r\d+_duration.*',
        r'^f_\d+_sig_strikes_.*',
        r'^f_\d+_takedown_.*',
        r'^f_\d+_submission_.*',
        r'^f_\d+_knockdowns.*',
        r'^winner$',
        r'^result$',
        r'^finish_.*',
    ]

    leaking_cols = []
    for col in df.columns:
        if col == 'winner_encoded':
            continue
        for pattern in leaking_patterns:
            if re.match(pattern, col):
                leaking_cols.append(col)
                break

    df_clean = df.drop(columns=leaking_cols)

    metadata_cols = ['event_date', 'event_name', 'event_location']
    target_col = 'winner_encoded'

    feature_cols = [
        col for col in df_clean.columns
        if col not in metadata_cols
        and col != target_col
        and df_clean[col].dtype in ['float64', 'int64', 'float32', 'int32']
    ]

    logger.info(f"✓ Total features: {len(feature_cols)}")
    logger.info(f"✓ Removed by patterns: {len(leaking_cols)}\n")

    # Audit each feature
    logger.info("="*80)
    logger.info("AUDITING FEATURES")
    logger.info("="*80)

    definite_leaks = []
    suspicious_features = []
    safe_features = []
    unknown_features = []

    for col in feature_cols:
        sample_vals = df_clean[col].dropna().head(5).tolist()
        is_leak, reason = audit_feature_for_leakage(col, sample_vals)

        if is_leak is True:
            definite_leaks.append((col, reason))
        elif is_leak is None:
            if "SUSPICIOUS" in reason:
                suspicious_features.append((col, reason))
            else:
                unknown_features.append((col, reason))
        else:
            safe_features.append((col, reason))

    # Report findings
    logger.info(f"\n{'='*80}")
    logger.info("AUDIT RESULTS")
    logger.info("="*80)

    logger.info(f"\n✓ SAFE features: {len(safe_features)}")
    logger.info(f"⚠️  SUSPICIOUS features: {len(suspicious_features)}")
    logger.info(f"❓ UNKNOWN features: {len(unknown_features)}")

    if definite_leaks:
        logger.error(f"\n🚨 DEFINITE LEAKS FOUND: {len(definite_leaks)}")
        logger.error("\nThese features MUST be removed:")
        for col, reason in definite_leaks[:20]:
            logger.error(f"  - {col}: {reason}")
        if len(definite_leaks) > 20:
            logger.error(f"  ... and {len(definite_leaks) - 20} more")
    else:
        logger.success("\n✓ NO DEFINITE LEAKS FOUND")

    if suspicious_features:
        logger.warning(f"\n⚠️  SUSPICIOUS FEATURES ({len(suspicious_features)}):")
        logger.warning("These need manual inspection:")
        for col, reason in suspicious_features[:20]:
            logger.warning(f"  - {col}: {reason}")
        if len(suspicious_features) > 20:
            logger.warning(f"  ... and {len(suspicious_features) - 20} more")

    if unknown_features:
        logger.info(f"\n❓ UNKNOWN FEATURES ({len(unknown_features)}):")
        logger.info("These need manual inspection:")
        for col, reason in unknown_features[:30]:
            logger.info(f"  - {col}")
        if len(unknown_features) > 30:
            logger.info(f"  ... and {len(unknown_features) - 30} more")

    # Sample some unknown features for inspection
    if unknown_features:
        logger.info("\n" + "="*80)
        logger.info("SAMPLING UNKNOWN FEATURES FOR INSPECTION")
        logger.info("="*80)

        sample_cols = [col for col, _ in unknown_features[:10]]
        sample_df = df_clean[sample_cols].head(10)

        for col in sample_cols:
            logger.info(f"\n{col}:")
            logger.info(f"  Sample values: {df_clean[col].dropna().head(5).tolist()}")
            logger.info(f"  Non-null: {df_clean[col].notna().sum()} / {len(df_clean)}")
            logger.info(f"  Mean: {df_clean[col].mean():.3f}, Std: {df_clean[col].std():.3f}")

    # Check for odds features
    logger.info("\n" + "="*80)
    logger.info("ODDS FEATURES CHECK")
    logger.info("="*80)

    odds_features = [col for col in feature_cols if 'odds' in col.lower()]
    logger.info(f"\nOdds-related features found: {len(odds_features)}")
    for col in odds_features:
        logger.info(f"  - {col}")
        logger.info(f"    Available in: {df_clean[col].notna().sum()} / {len(df_clean)} fights ({df_clean[col].notna().mean():.1%})")

    # Final verdict
    logger.info("\n" + "="*80)
    logger.info("FINAL VERDICT")
    logger.info("="*80)

    if definite_leaks:
        logger.error(f"\n🚨 MODEL HAS DATA LEAKAGE!")
        logger.error(f"Found {len(definite_leaks)} definite leaks that must be removed")
        logger.error("Results are NOT trustworthy until leaks are fixed")
    elif suspicious_features:
        logger.warning(f"\n⚠️  MODEL NEEDS REVIEW")
        logger.warning(f"Found {len(suspicious_features)} suspicious features")
        logger.warning("Manual inspection required before trusting results")
    else:
        logger.success(f"\n✓ MODEL APPEARS LEAK-FREE")
        logger.info(f"All {len(feature_cols)} features passed automated checks")
        logger.info("Betting odds are included (valid pre-fight data)")

        if unknown_features:
            logger.info(f"\nNote: {len(unknown_features)} features need manual inspection to be 100% certain")

    logger.info("\n" + "="*80)
    logger.info("AUDIT COMPLETE")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
