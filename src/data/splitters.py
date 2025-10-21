"""
Temporal data splitting for time-series UFC data.

CRITICAL: Never shuffle time-series data - preserves temporal ordering.
"""

import pandas as pd
from typing import NamedTuple
from loguru import logger

from src.utils.config import get_config


class DataSplit(NamedTuple):
    """Container for train/val/test splits"""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class TemporalSplitter:
    """
    Split UFC data by date (temporal split).

    CRITICAL: Maintains strict temporal ordering:
    - Train: All fights before val_start_date
    - Val: Fights between val_start_date and test_start_date
    - Test: Fights after test_start_date (ONE-TIME HOLDOUT)

    Example:
        splitter = TemporalSplitter()
        split = splitter.split(df)
        print(f"Train: {len(split.train)} fights")
        print(f"Val: {len(split.val)} fights")
        print(f"Test: {len(split.test)} fights")
    """

    def __init__(
        self,
        val_start_date: str = None,
        test_start_date: str = None,
        date_column: str = None
    ):
        """
        Initialize temporal splitter.

        Args:
            val_start_date: Start date for validation set (default: from config)
            test_start_date: Start date for test set (default: from config)
            date_column: Date column name (default: from config)
        """
        config = get_config()

        self.val_start_date = pd.to_datetime(val_start_date or config.splits.val_start_date)
        self.test_start_date = pd.to_datetime(test_start_date or config.splits.test_start_date)
        self.date_column = date_column or config.splits.date_column

        logger.info("="*80)
        logger.info("TEMPORAL SPLITTER INITIALIZED")
        logger.info("="*80)
        logger.info(f"\nDate column: {self.date_column}")
        logger.info(f"Train: < {self.val_start_date.date()}")
        logger.info(f"Val: {self.val_start_date.date()} to {self.test_start_date.date()}")
        logger.info(f"Test: >= {self.test_start_date.date()}")
        logger.info("="*80 + "\n")

    def split(self, df: pd.DataFrame) -> DataSplit:
        """
        Split data into train/val/test.

        Args:
            df: DataFrame with date column

        Returns:
            DataSplit(train, val, test)
        """
        logger.info("="*80)
        logger.info("SPLITTING DATA (TEMPORAL)")
        logger.info("="*80)

        # Ensure date column is datetime
        if df[self.date_column].dtype != 'datetime64[ns]':
            df[self.date_column] = pd.to_datetime(df[self.date_column])

        # Split by date
        train_mask = df[self.date_column] < self.val_start_date
        val_mask = (df[self.date_column] >= self.val_start_date) & \
                   (df[self.date_column] < self.test_start_date)
        test_mask = df[self.date_column] >= self.test_start_date

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()

        # Log split sizes
        logger.info(f"\nTotal fights: {len(df):,}")
        logger.info(f"\nTrain: {len(train_df):,} fights ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"  Date range: {train_df[self.date_column].min().date()} to {train_df[self.date_column].max().date()}")

        logger.info(f"\nVal: {len(val_df):,} fights ({len(val_df)/len(df)*100:.1f}%)")
        logger.info(f"  Date range: {val_df[self.date_column].min().date()} to {val_df[self.date_column].max().date()}")

        logger.info(f"\nTest: {len(test_df):,} fights ({len(test_df)/len(df)*100:.1f}%)")
        if len(test_df) > 0:
            logger.info(f"  Date range: {test_df[self.date_column].min().date()} to {test_df[self.date_column].max().date()}")
        else:
            logger.warning("  ⚠️  No test data (dates might be in future)")

        # Validation checks
        if len(train_df) == 0:
            logger.error("\n❌ No training data!")
            raise ValueError("Training set is empty")

        if len(val_df) == 0:
            logger.warning("\n⚠️  No validation data!")

        # Check for temporal overlap
        if len(train_df) > 0 and len(val_df) > 0:
            train_max = train_df[self.date_column].max()
            val_min = val_df[self.date_column].min()
            if train_max >= val_min:
                logger.error(f"\n❌ TEMPORAL LEAKAGE: Train max date {train_max.date()} >= Val min date {val_min.date()}")
                raise ValueError("Temporal leakage detected between train and val")

        if len(val_df) > 0 and len(test_df) > 0:
            val_max = val_df[self.date_column].max()
            test_min = test_df[self.date_column].min()
            if val_max >= test_min:
                logger.error(f"\n❌ TEMPORAL LEAKAGE: Val max date {val_max.date()} >= Test min date {test_min.date()}")
                raise ValueError("Temporal leakage detected between val and test")

        logger.success(f"\n✓ Temporal split complete (no overlap)")
        logger.info("="*80 + "\n")

        return DataSplit(train=train_df, val=val_df, test=test_df)


if __name__ == "__main__":
    # Test temporal splitter
    from src.data.loaders import load_ufc_data

    logger.info("Testing temporal splitter...\n")

    # Load data
    df = load_ufc_data(max_rows=5000)

    # Split data
    splitter = TemporalSplitter()
    split = splitter.split(df)

    logger.success("✓ Temporal splitter test passed!")
    logger.info(f"\nSplit summary:")
    logger.info(f"  Train: {len(split.train):,} fights")
    logger.info(f"  Val: {len(split.val):,} fights")
    logger.info(f"  Test: {len(split.test):,} fights")
