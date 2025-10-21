"""
Advanced preprocessing strategies with leak-free imputation.

Key Principle:
- Different feature types require different imputation strategies
- Physical stats (height, reach) → Median (represents typical fighter)
- Rolling stats (last N fights) → Zero (correct for debut fighters)
- Career stats → Median (represents typical performance)
- Create missingness indicators for additional signal
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from loguru import logger


class FeatureTypeImputationStrategy:
    """
    Intelligent imputation based on feature semantics.

    Different feature types have different missing value meanings:
    - Missing physical stat → Fighter data not available → Use population median
    - Missing rolling stat → Fighter has < N previous fights → Use 0 (correct interpretation)
    - Missing career stat → Data scraping issue → Use population median

    Example:
        imputer = FeatureTypeImputationStrategy()
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)
    """

    def __init__(self, create_indicators: bool = True):
        """
        Initialize imputation strategy.

        Args:
            create_indicators: If True, create binary features indicating missingness
        """
        self.create_indicators = create_indicators

        # Imputers for different feature types
        self.physical_imputer = SimpleImputer(strategy='median')
        self.career_imputer = SimpleImputer(strategy='median')
        self.odds_imputer = SimpleImputer(strategy='median')
        self.rolling_imputer = SimpleImputer(strategy='constant', fill_value=0)

        # Feature group mappings (learned during fit)
        self.feature_groups = {
            'physical': [],
            'career': [],
            'rolling': [],
            'odds': [],
            'other': []
        }

        # Indicator column names
        self.indicator_cols = []

        self.fitted = False

    def _categorize_features(self, columns: List[str]) -> Dict[str, List[str]]:
        """
        Categorize features into semantic groups.

        Args:
            columns: List of feature column names

        Returns:
            Dictionary mapping category -> list of column names
        """
        groups = {
            'physical': [],
            'career': [],
            'rolling': [],
            'odds': [],
            'other': []
        }

        for col in columns:
            col_lower = col.lower()

            # Physical attributes (height, reach, weight, age)
            if any(kw in col_lower for kw in ['height', 'reach', 'weight', 'age']):
                groups['physical'].append(col)

            # Betting odds
            elif any(kw in col_lower for kw in ['odds', 'prob', 'implied']):
                groups['odds'].append(col)

            # Rolling statistics (e.g., strikes_5_f_1 = avg strikes over last 5 fights)
            elif any(f'_{i}_' in col for i in range(3, 16)):
                groups['rolling'].append(col)

            # Career statistics (e.g., SlpM, Str_Acc, TD_Avg)
            elif any(kw in col_lower for kw in ['slpm', 'str_acc', 'sapm', 'str_def',
                                                  'td_avg', 'td_acc', 'td_def', 'sub_avg',
                                                  'fighter_w', 'fighter_l', 'fighter_d']):
                groups['career'].append(col)

            # Everything else
            else:
                groups['other'].append(col)

        return groups

    def fit(self, X: pd.DataFrame) -> 'FeatureTypeImputationStrategy':
        """
        Fit imputers on training data.

        CRITICAL: Only call this on TRAINING data, never on validation/test!

        Args:
            X: Training feature DataFrame

        Returns:
            self (fitted imputer)
        """
        logger.info("="*80)
        logger.info("FEATURE-TYPE-SPECIFIC IMPUTATION")
        logger.info("="*80)

        # Categorize features
        self.feature_groups = self._categorize_features(X.columns.tolist())

        # Log feature group sizes
        logger.info("\nFeature categorization:")
        for group_name, features in self.feature_groups.items():
            logger.info(f"  {group_name:12s}: {len(features):4d} features")

        # Fit imputers on each group (TRAINING DATA ONLY!)
        logger.info("\nFitting imputers (TRAINING DATA ONLY):")

        if len(self.feature_groups['physical']) > 0:
            logger.info(f"  Physical features: Median imputation")
            # Filter out non-numeric columns
            physical_numeric = [c for c in self.feature_groups['physical'] if X[c].dtype in ['float64', 'int64', 'float32', 'int32']]
            if len(physical_numeric) > 0:
                self.physical_imputer.fit(X[physical_numeric])
                self.feature_groups['physical'] = physical_numeric  # Update to numeric only

        if len(self.feature_groups['career']) > 0:
            logger.info(f"  Career features: Median imputation")
            # Filter out non-numeric columns
            career_numeric = [c for c in self.feature_groups['career'] if X[c].dtype in ['float64', 'int64', 'float32', 'int32']]
            if len(career_numeric) > 0:
                self.career_imputer.fit(X[career_numeric])
                self.feature_groups['career'] = career_numeric  # Update to numeric only

        if len(self.feature_groups['odds']) > 0:
            logger.info(f"  Odds features: Median imputation")
            # Filter out non-numeric columns
            odds_numeric = [c for c in self.feature_groups['odds'] if X[c].dtype in ['float64', 'int64', 'float32', 'int32']]
            if len(odds_numeric) > 0:
                self.odds_imputer.fit(X[odds_numeric])
                self.feature_groups['odds'] = odds_numeric  # Update to numeric only

        if len(self.feature_groups['rolling']) > 0:
            logger.info(f"  Rolling features: Zero-fill (correct for debuts)")
            self.rolling_imputer.fit(X[self.feature_groups['rolling']])

        self.fitted = True
        logger.success("✓ Imputation strategy fitted\n")

        return self

    def transform(self, X: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Apply fitted imputation to data.

        Args:
            X: Feature DataFrame to impute
            verbose: Print detailed transformation logs

        Returns:
            Imputed DataFrame with same columns + optional indicators
        """
        if not self.fitted:
            raise ValueError("Imputer not fitted. Call .fit() first.")

        X_imputed = X.copy()

        # Count missing values before imputation
        missing_before = X_imputed.isna().sum().sum()

        if verbose:
            logger.info(f"Imputing {missing_before:,} missing values...")

        # Apply imputation to each feature group
        if len(self.feature_groups['physical']) > 0:
            X_imputed[self.feature_groups['physical']] = self.physical_imputer.transform(
                X_imputed[self.feature_groups['physical']]
            )

        if len(self.feature_groups['career']) > 0:
            X_imputed[self.feature_groups['career']] = self.career_imputer.transform(
                X_imputed[self.feature_groups['career']]
            )

        if len(self.feature_groups['odds']) > 0:
            X_imputed[self.feature_groups['odds']] = self.odds_imputer.transform(
                X_imputed[self.feature_groups['odds']]
            )

        if len(self.feature_groups['rolling']) > 0:
            X_imputed[self.feature_groups['rolling']] = self.rolling_imputer.transform(
                X_imputed[self.feature_groups['rolling']]
            )

        # Create missingness indicators (additional signal)
        if self.create_indicators:
            for col in X.columns:
                if X[col].isna().any():
                    indicator_name = f"{col}_missing"
                    X_imputed[indicator_name] = X[col].isna().astype(int)

                    if indicator_name not in self.indicator_cols:
                        self.indicator_cols.append(indicator_name)

        # Verify no missing values remain
        missing_after = X_imputed.isna().sum().sum()

        if missing_after > 0:
            logger.warning(f"⚠️  {missing_after} missing values remain after imputation!")
        elif verbose:
            logger.success(f"✓ All {missing_before:,} missing values imputed")
            if self.create_indicators:
                logger.info(f"✓ Created {len(self.indicator_cols)} missingness indicators")

        return X_imputed

    def fit_transform(self, X: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Fit on training data and transform it.

        Args:
            X: Training feature DataFrame
            verbose: Print logs

        Returns:
            Imputed training DataFrame
        """
        self.fit(X)
        return self.transform(X, verbose=verbose)

    def get_feature_info(self) -> pd.DataFrame:
        """
        Get summary of feature groups and imputation strategies.

        Returns:
            DataFrame with feature group information
        """
        info = []

        for group_name, features in self.feature_groups.items():
            if len(features) == 0:
                continue

            # Determine imputation strategy
            if group_name == 'physical':
                strategy = 'Median (typical fighter)'
            elif group_name == 'career':
                strategy = 'Median (typical performance)'
            elif group_name == 'rolling':
                strategy = 'Zero (correct for debuts)'
            elif group_name == 'odds':
                strategy = 'Median (typical odds)'
            else:
                strategy = 'None (no missing values expected)'

            info.append({
                'Feature Group': group_name,
                'Count': len(features),
                'Imputation Strategy': strategy,
                'Example Features': ', '.join(features[:3])
            })

        return pd.DataFrame(info)


if __name__ == "__main__":
    # Test with synthetic data
    logger.info("Testing FeatureTypeImputationStrategy with synthetic data...")

    # Create synthetic dataset with missing values
    np.random.seed(42)

    data = {
        # Physical features (should use median)
        'f_1_fighter_height_cm': [180, 175, np.nan, 185, 190],
        'f_1_fighter_reach_cm': [180, np.nan, 185, 190, 175],
        'f_1_fighter_weight_lbs': [170, 165, 175, np.nan, 180],

        # Career stats (should use median)
        'f_1_fighter_SlpM': [5.2, 4.8, np.nan, 6.1, 5.5],
        'f_1_fighter_Str_Acc': [0.45, 0.50, 0.48, np.nan, 0.52],

        # Rolling stats (should use 0 for debuts)
        'strikes_3_f_1': [100, 95, np.nan, 105, 98],  # Missing = debut fighter
        'wins_5_f_1': [3, 4, np.nan, 2, 5],

        # Odds (should use median)
        'f_1_odds': [1.8, 2.1, np.nan, 1.5, 2.5],
    }

    X_train = pd.DataFrame(data)

    logger.info("\nOriginal data with missing values:")
    logger.info(f"\n{X_train}")
    logger.info(f"\nMissing value counts:\n{X_train.isna().sum()}")

    # Create and fit imputer
    imputer = FeatureTypeImputationStrategy(create_indicators=True)
    X_imputed = imputer.fit_transform(X_train, verbose=True)

    logger.info("\nImputed data:")
    logger.info(f"\n{X_imputed}")

    logger.info("\nFeature group info:")
    info = imputer.get_feature_info()
    logger.info(f"\n{info.to_string(index=False)}")

    # Verify imputation strategies
    logger.info("\n" + "="*80)
    logger.info("VERIFICATION")
    logger.info("="*80)

    # Check rolling stats imputed with 0
    rolling_cols = ['strikes_3_f_1', 'wins_5_f_1']
    for col in rolling_cols:
        imputed_value = X_imputed.loc[2, col]  # Row 2 had NaN
        logger.info(f"{col}: Imputed with {imputed_value} (expected 0)")
        assert imputed_value == 0, f"Rolling stat should be imputed with 0, got {imputed_value}"

    # Check physical/career stats imputed with median
    physical_cols = ['f_1_fighter_height_cm', 'f_1_fighter_reach_cm']
    for col in physical_cols:
        if col in X_train.columns and X_train[col].isna().any():
            median_val = X_train[col].median()
            row_with_nan = X_train[col].isna().idxmax()
            imputed_value = X_imputed.loc[row_with_nan, col]
            logger.info(f"{col}: Imputed with {imputed_value:.1f} (median: {median_val:.1f})")

    # Check indicators created
    indicator_cols = [col for col in X_imputed.columns if '_missing' in col]
    logger.info(f"\n✓ Created {len(indicator_cols)} missingness indicators:")
    for col in indicator_cols[:5]:
        logger.info(f"  - {col}")

    logger.success("\n✓ FeatureTypeImputationStrategy test passed!")
