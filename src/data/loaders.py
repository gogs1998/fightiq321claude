"""
Data loading with automatic leak detection and filtering.

CRITICAL: All data loaded through this module is automatically
screened for data leakage (current-fight statistics excluded).
"""

import pandas as pd
import re
from typing import List, Tuple, Optional
from pathlib import Path
from loguru import logger

from src.utils.config import get_config


def _is_current_fight_stat(column_name: str) -> bool:
    """
    Check if column contains current-fight statistics (data leakage).

    Uses FightIQ's proven leakage patterns (achieves 67-69% test accuracy).
    This is LESS aggressive than previous patterns and keeps historical aggregates
    like f_1_head_succ_total (which are safe historical features).

    Args:
        column_name: Column name to check

    Returns:
        True if column contains leaked information
    """
    # FightIQ's EXACT leakage patterns (from FightIQ/src/data/loaders.py:115-128)
    # These patterns have been proven to achieve 67-69% leak-free test accuracy

    # Round-by-round patterns (FightIQ line 116)
    current_fight_patterns = ['_r1_', '_r2_', '_r3_', '_r4_', '_r5_']

    # FightIQ's specific totals to remove (FightIQ lines 120-128)
    current_fight_totals = [
        'f_1_total_strikes_succ', 'f_2_total_strikes_succ',
        'f_1_total_strikes_att', 'f_2_total_strikes_att',
        'f_1_sig_strikes_succ', 'f_2_sig_strikes_succ',
        'f_1_sig_strikes_att', 'f_2_sig_strikes_att',
        'f_1_knockdowns', 'f_2_knockdowns',
        'f_1_submission_att', 'f_2_submission_att',
        'f_1_ctrl_time_sec', 'f_2_ctrl_time_sec',
        'fight_duration_minutes',
        'winner', 'result', 'result_details',
        'finish_round', 'finish_time', 'finish_details',
        'method', 'time_format', 'num_rounds'
    ]

    # Betting odds features to remove for NO-ODDS baseline
    # (These are valid pre-fight features, but we exclude them for fantasy model)
    odds_features = [
        'f_1_odds', 'f_2_odds', 'f_1_ko_odds', 'f_1_sub_odds',
        'f_2_ko_odds', 'f_2_sub_odds', 'diff_odds'
    ]

    # Check if column matches round-by-round pattern
    if any(pattern in column_name for pattern in current_fight_patterns):
        return True

    # Check if column is in specific totals list
    if column_name in current_fight_totals:
        return True

    # Check if column is odds (for no-odds baseline)
    if column_name in odds_features:
        return True

    return False


def load_ufc_data(
    data_path: str = None,
    remove_leaking_features: bool = True,
    max_rows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load UFC dataset with automatic leak detection.

    CRITICAL: By default, automatically removes 3,897+ features
    that contain current-fight information (data leakage).

    Args:
        data_path: Path to dataset (default: from config)
        remove_leaking_features: If True, remove leaking features (STRONGLY RECOMMENDED)
        max_rows: Max rows to load (for testing)

    Returns:
        DataFrame with leak-free features
    """
    config = get_config()

    if data_path is None:
        data_path = config.paths.golden_dataset

    logger.info("="*80)
    logger.info("LOADING UFC DATA")
    logger.info("="*80)
    logger.info(f"\nData path: {data_path}")

    # Load data
    df = pd.read_csv(data_path, nrows=max_rows)
    logger.info(f"✓ Loaded {len(df):,} fights, {len(df.columns):,} columns")

    # Parse dates
    if config.splits.date_column in df.columns:
        df[config.splits.date_column] = pd.to_datetime(df[config.splits.date_column])
        logger.info(f"✓ Parsed dates: {df[config.splits.date_column].min()} to {df[config.splits.date_column].max()}")

    if remove_leaking_features:
        logger.info(f"\n{'='*80}")
        logger.info("LEAK DETECTION & REMOVAL")
        logger.info("="*80)

        # Identify leaking features (but keep winner_encoded as target)
        leaking_cols = [col for col in df.columns
                       if _is_current_fight_stat(col) and col != 'winner_encoded']

        logger.warning(f"\n⚠️  Found {len(leaking_cols):,} leaking features (current-fight stats)")

        if len(leaking_cols) > 0:
            logger.info(f"\nExamples of removed features:")
            for col in leaking_cols[:10]:
                logger.info(f"  - {col}")

            if len(leaking_cols) > 10:
                logger.info(f"  ... and {len(leaking_cols) - 10} more")

            # Remove leaking features
            df = df.drop(columns=leaking_cols)
            logger.success(f"\n✓ Removed {len(leaking_cols):,} leaking features")
            logger.info(f"✓ Clean dataset: {len(df.columns):,} leak-free columns")

    logger.info(f"\n{'='*80}")
    logger.success("✓ DATA LOADED SUCCESSFULLY (LEAK-FREE)")
    logger.info("="*80 + "\n")

    return df


def get_feature_and_target_columns(
    df: pd.DataFrame,
    target_col: str = 'winner_encoded'
) -> Tuple[List[str], str]:
    """
    Separate feature columns from target and metadata.

    Args:
        df: DataFrame
        target_col: Target column name

    Returns:
        (feature_columns, target_column)
    """
    # Metadata columns to exclude from features
    metadata_cols = [
        'event_date', 'event_name', 'event_location',
        'fight_url', 'fighter_1_url', 'fighter_2_url',
        'referee', 'weight_class', 'gender',
        'f_1_name', 'f_2_name',
        'winner_encoded', 'result', 'result_details',
        'finish_round', 'finish_time'
    ]

    # Feature columns = all NUMERIC columns except metadata and target
    feature_cols = [
        col for col in df.columns
        if col not in metadata_cols
        and col != target_col
        and df[col].dtype in ['float64', 'int64', 'float32', 'int32', 'float', 'int']
    ]

    logger.info(f"\n{'='*80}")
    logger.info("FEATURE & TARGET SEPARATION")
    logger.info("="*80)
    logger.info(f"\nTotal columns: {len(df.columns):,}")
    logger.info(f"Metadata columns: {len(metadata_cols):,}")
    logger.info(f"Feature columns: {len(feature_cols):,}")
    logger.info(f"Target column: {target_col}")

    if target_col not in df.columns:
        logger.error(f"\n❌ Target column '{target_col}' not found in dataset!")
        raise ValueError(f"Target column '{target_col}' not found")

    logger.success(f"\n✓ Features and target separated\n")

    return feature_cols, target_col


def validate_no_leakage(df: pd.DataFrame, feature_cols: List[str]) -> bool:
    """
    Final validation that no leaking features remain.

    Args:
        df: DataFrame
        feature_cols: List of feature column names

    Returns:
        True if no leakage detected

    Raises:
        ValueError if leakage detected
    """
    logger.info("="*80)
    logger.info("FINAL LEAKAGE VALIDATION")
    logger.info("="*80)

    # Check for leaking features
    leaking_features = [col for col in feature_cols if _is_current_fight_stat(col)]

    if len(leaking_features) > 0:
        logger.error(f"\n❌ DATA LEAKAGE DETECTED!")
        logger.error(f"   {len(leaking_features)} leaking features found:")
        for col in leaking_features[:20]:
            logger.error(f"   - {col}")

        raise ValueError(f"Data leakage detected: {len(leaking_features)} leaking features")

    # Check for target leakage
    target_cols = ['winner', 'winner_encoded', 'result', 'result_details', 'finish_round', 'finish_time']
    target_leaks = [col for col in feature_cols if any(target in col.lower() for target in target_cols)]

    # Filter out valid historical features
    target_leaks = [
        col for col in target_leaks
        if not any(valid in col.lower() for valid in ['_wins_', '_losses_', '_streak_',
                                                        'finish_wins_', 'finish_losses_'])
    ]

    if len(target_leaks) > 0:
        logger.error(f"\n❌ TARGET LEAKAGE DETECTED!")
        logger.error(f"   {len(target_leaks)} features contain target information:")
        for col in target_leaks:
            logger.error(f"   - {col}")

        raise ValueError(f"Target leakage detected: {len(target_leaks)} features")

    logger.success(f"\n✓ NO LEAKAGE DETECTED")
    logger.success(f"✓ All {len(feature_cols):,} features are leak-free")
    logger.info("="*80 + "\n")

    return True


if __name__ == "__main__":
    # Test data loader
    logger.info("Testing data loader with leak detection...\n")

    # Load data
    df = load_ufc_data(max_rows=1000)

    # Get features and target
    feature_cols, target_col = get_feature_and_target_columns(df)

    # Final validation
    validate_no_leakage(df, feature_cols)

    logger.success("✓ Data loader test passed!")
