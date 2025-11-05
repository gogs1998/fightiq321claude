"""
Comprehensive Data Leakage Audit Script

Performs multiple tests to detect data leakage:
1. Feature-level analysis
2. Temporal contamination check
3. Shuffle test (random split should perform worse)
4. Feature importance analysis
5. Correlation with target

Run this before any model training to ensure data integrity.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def check_suspicious_features(df: pd.DataFrame, target_col: str = 'target') -> list:
    """
    Check for features with suspiciously high correlation with target

    Args:
        df: Dataset
        target_col: Target column name

    Returns:
        List of suspicious features
    """
    logger.info("="*80)
    logger.info("SUSPICIOUS FEATURE DETECTION")
    logger.info("="*80)

    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found")
        return []

    # Calculate correlation with target for numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_col]

    correlations = {}
    for col in numeric_cols:
        if df[col].nunique() > 1:  # Skip constant columns
            corr = abs(df[col].corr(df[target_col]))
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by correlation (descending)
    sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    # Flag features with correlation > 0.5 (suspiciously high)
    suspicious = []
    threshold = 0.5

    logger.info(f"\nTop 20 features by correlation with target:")
    logger.info("-" * 80)

    for i, (feature, corr) in enumerate(sorted_corrs[:20]):
        status = "⚠️  SUSPICIOUS" if corr > threshold else "✓"
        logger.info(f"{i+1:2d}. {feature:50s} {corr:.4f} {status}")

        if corr > threshold:
            suspicious.append((feature, corr))

    if suspicious:
        logger.warning(f"\n⚠️  Found {len(suspicious)} features with correlation > {threshold}")
        logger.warning("These features may contain leakage and should be investigated:")
        for feature, corr in suspicious:
            logger.warning(f"  - {feature}: {corr:.4f}")
    else:
        logger.success(f"\n✓ No features with correlation > {threshold}")

    return suspicious


def shuffle_test(df: pd.DataFrame, target_col: str = 'target', date_col: str = 'event_date') -> dict:
    """
    Shuffle test: Random split should perform WORSE than temporal split

    If random split performs similarly to or better than temporal split,
    there may be leakage.

    Args:
        df: Dataset
        target_col: Target column
        date_col: Date column

    Returns:
        Dict with test results
    """
    logger.info("\n" + "="*80)
    logger.info("SHUFFLE TEST (Random vs Temporal Split)")
    logger.info("="*80)
    logger.info("If random split performs similar to temporal split, LEAKAGE SUSPECTED")

    # Prepare data
    feature_cols = [col for col in df.columns
                   if col not in [target_col, date_col, 'f_1_name', 'f_2_name',
                                  'event_name', 'event_location']]
    feature_cols = [col for col in feature_cols if df[col].dtype in [np.number]]

    X = df[feature_cols].fillna(0)
    y = df[target_col]

    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # TEMPORAL SPLIT (correct methodology)
    logger.info("\n1. TEMPORAL SPLIT (80/20)")
    split_date = df[date_col].quantile(0.8)
    train_mask = df[date_col] < split_date
    test_mask = df[date_col] >= split_date

    X_train_temporal = X[train_mask]
    X_test_temporal = X[test_mask]
    y_train_temporal = y[train_mask]
    y_test_temporal = y[test_mask]

    # Train model
    rf_temporal = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_temporal.fit(X_train_temporal, y_train_temporal)

    y_pred_temporal = rf_temporal.predict(X_test_temporal)
    y_prob_temporal = rf_temporal.predict_proba(X_test_temporal)[:, 1]

    acc_temporal = accuracy_score(y_test_temporal, y_pred_temporal)
    auc_temporal = roc_auc_score(y_test_temporal, y_prob_temporal)

    logger.info(f"  Train: {len(X_train_temporal)} fights")
    logger.info(f"  Test: {len(X_test_temporal)} fights")
    logger.info(f"  Accuracy: {acc_temporal:.4f}")
    logger.info(f"  AUC: {auc_temporal:.4f}")

    # RANDOM SPLIT (should perform WORSE if no leakage)
    logger.info("\n2. RANDOM SPLIT (80/20)")
    X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_random = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_random.fit(X_train_random, y_train_random)

    y_pred_random = rf_random.predict(X_test_random)
    y_prob_random = rf_random.predict_proba(X_test_random)[:, 1]

    acc_random = accuracy_score(y_test_random, y_pred_random)
    auc_random = roc_auc_score(y_test_random, y_prob_random)

    logger.info(f"  Train: {len(X_train_random)} fights")
    logger.info(f"  Test: {len(X_test_random)} fights")
    logger.info(f"  Accuracy: {acc_random:.4f}")
    logger.info(f"  AUC: {auc_random:.4f}")

    # ANALYSIS
    logger.info("\n" + "-"*80)
    logger.info("SHUFFLE TEST RESULTS")
    logger.info("-"*80)

    acc_diff = acc_random - acc_temporal
    auc_diff = auc_random - auc_temporal

    logger.info(f"Random accuracy: {acc_random:.4f}")
    logger.info(f"Temporal accuracy: {acc_temporal:.4f}")
    logger.info(f"Difference: {acc_diff:+.4f} ({acc_diff/acc_temporal*100:+.1f}%)")

    logger.info(f"\nRandom AUC: {auc_random:.4f}")
    logger.info(f"Temporal AUC: {auc_temporal:.4f}")
    logger.info(f"Difference: {auc_diff:+.4f} ({auc_diff/auc_temporal*100:+.1f}%)")

    # VERDICT
    logger.info("\n" + "="*80)
    logger.info("VERDICT")
    logger.info("="*80)

    leakage_suspected = False

    if acc_diff > 0.02:  # Random is >2% better
        logger.error("⚠️  LEAKAGE SUSPECTED: Random split significantly outperforms temporal")
        logger.error(f"   Random split is {acc_diff:.1%} better - this should not happen!")
        leakage_suspected = True
    elif abs(acc_diff) < 0.01:  # Too similar (<1% difference)
        logger.warning("⚠️  POSSIBLE LEAKAGE: Random and temporal splits too similar")
        logger.warning(f"   Expected temporal to be 2-5% worse, but only {-acc_diff:.1%} difference")
        leakage_suspected = True
    else:
        logger.success("✓ PASS: Temporal split performs as expected")
        logger.success(f"   Temporal is {-acc_diff:.1%} worse than random (healthy)")

    return {
        'temporal_accuracy': acc_temporal,
        'random_accuracy': acc_random,
        'temporal_auc': auc_temporal,
        'random_auc': auc_random,
        'leakage_suspected': leakage_suspected
    }


def audit_new_features(df: pd.DataFrame, new_feature_patterns: list) -> dict:
    """
    Audit newly added features for leakage

    Args:
        df: Dataset with new features
        new_feature_patterns: List of patterns to match new features

    Returns:
        Dict with audit results
    """
    logger.info("\n" + "="*80)
    logger.info("NEW FEATURE LEAKAGE AUDIT")
    logger.info("="*80)

    issues = []

    for pattern in new_feature_patterns:
        matching_cols = [col for col in df.columns if pattern in col]

        if not matching_cols:
            continue

        logger.info(f"\nAuditing features matching '{pattern}':")
        logger.info(f"Found {len(matching_cols)} features")

        for col in matching_cols:
            # Check 1: Does name suggest current-fight data?
            suspicious_keywords = ['_r1_', '_r2_', '_r3_', '_current_', '_total_strikes',
                                  '_knockdown', '_finish_', 'winner', 'result']

            if any(kw in col.lower() for kw in suspicious_keywords):
                issues.append({
                    'feature': col,
                    'issue': 'Suspicious name pattern',
                    'severity': 'HIGH'
                })
                logger.error(f"  ⚠️  {col}: Suspicious name pattern")

            # Check 2: Constant or near-constant values (might be computed wrong)
            if df[col].dtype in [np.number]:
                if df[col].nunique() <= 2:
                    issues.append({
                        'feature': col,
                        'issue': 'Constant or binary',
                        'severity': 'MEDIUM'
                    })
                    logger.warning(f"  ⚠️  {col}: Only {df[col].nunique()} unique values")

    if issues:
        logger.warning(f"\n⚠️  Found {len(issues)} potential issues with new features")
    else:
        logger.success("\n✓ No obvious issues found in new features")

    return {'issues': issues}


def check_temporal_contamination(df: pd.DataFrame, date_col: str = 'event_date') -> bool:
    """
    Check if features might contain information from future fights

    Args:
        df: Dataset
        date_col: Date column

    Returns:
        True if contamination detected
    """
    logger.info("\n" + "="*80)
    logger.info("TEMPORAL CONTAMINATION CHECK")
    logger.info("="*80)

    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)

    # For each fighter, check if their stats are monotonically increasing
    # (which they should be if computed correctly - career stats only go up)

    logger.info("\nChecking fighter career stats for temporal consistency...")

    # Sample a few fighters
    if 'f_1_name' in df.columns:
        sample_fighters = df['f_1_name'].value_counts().head(5).index.tolist()

        for fighter in sample_fighters:
            fighter_fights = df[df['f_1_name'] == fighter].sort_values(date_col)

            if len(fighter_fights) < 3:
                continue

            # Check if wins column is monotonically increasing
            if 'f_1_fighter_w' in df.columns:
                wins = fighter_fights['f_1_fighter_w'].values

                # Wins should only increase or stay same (never decrease)
                if np.any(np.diff(wins) < 0):
                    logger.error(f"⚠️  {fighter}: Win count DECREASES over time - LEAKAGE!")
                    return True
                else:
                    logger.info(f"  ✓ {fighter}: Win progression looks valid")

    logger.success("\n✓ No temporal contamination detected")
    return False


def main():
    """Run comprehensive leakage audit"""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive data leakage audit")
    parser.add_argument('--data', default='data/fightiq_golden_dataset.csv', help='Path to dataset')
    parser.add_argument('--target', default='target', help='Target column name')
    parser.add_argument('--date', default='event_date', help='Date column name')
    parser.add_argument('--new-features', nargs='*', default=['win_streak', 'momentum', 'form_trend', 'archetype'],
                       help='Patterns for new features to audit')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("FIGHTIQ COMPREHENSIVE DATA LEAKAGE AUDIT")
    logger.info("="*80 + "\n")

    # Load data
    logger.info(f"Loading data from: {args.data}")
    df = pd.read_csv(args.data)
    logger.success(f"✓ Loaded {len(df)} fights, {len(df.columns)} features\n")

    # Run audits
    audit_results = {}

    # 1. Check suspicious features
    suspicious = check_suspicious_features(df, args.target)
    audit_results['suspicious_features'] = suspicious

    # 2. Shuffle test
    shuffle_results = shuffle_test(df, args.target, args.date)
    audit_results['shuffle_test'] = shuffle_results

    # 3. Audit new features
    if args.new_features:
        new_feature_audit = audit_new_features(df, args.new_features)
        audit_results['new_features'] = new_feature_audit

    # 4. Check temporal contamination
    contamination = check_temporal_contamination(df, args.date)
    audit_results['temporal_contamination'] = contamination

    # FINAL VERDICT
    logger.info("\n" + "="*80)
    logger.info("FINAL AUDIT VERDICT")
    logger.info("="*80)

    issues_found = []

    if len(suspicious) > 0:
        issues_found.append(f"{len(suspicious)} suspicious features (correlation > 0.5)")

    if shuffle_results['leakage_suspected']:
        issues_found.append("Shuffle test failed (random split too good)")

    if audit_results.get('new_features', {}).get('issues'):
        issues_found.append(f"{len(audit_results['new_features']['issues'])} issues in new features")

    if contamination:
        issues_found.append("Temporal contamination detected")

    if issues_found:
        logger.error("\n❌ DATA LEAKAGE DETECTED")
        logger.error("Issues found:")
        for issue in issues_found:
            logger.error(f"  - {issue}")

        logger.info("\nRECOMMENDED ACTIONS:")
        logger.info("1. Review suspicious features manually")
        logger.info("2. Check feature engineering code for current-fight data")
        logger.info("3. Verify temporal ordering in feature computation")
        logger.info("4. Consider removing high-correlation features")

        return False

    else:
        logger.success("\n✓ NO LEAKAGE DETECTED")
        logger.success("All audits passed successfully")
        logger.info(f"\nBaseline performance (temporal split):")
        logger.info(f"  Accuracy: {shuffle_results['temporal_accuracy']:.4f}")
        logger.info(f"  AUC: {shuffle_results['temporal_auc']:.4f}")
        logger.info("\nThis is a realistic estimate of model performance")

        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
