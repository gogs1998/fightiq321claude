"""
Demo: Run leakage audit with synthetic data to demonstrate methodology

This creates a small synthetic dataset to test the audit script
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger


def create_synthetic_ufc_data(n_fights=1000, seed=42):
    """
    Create synthetic UFC fight data for testing

    Args:
        n_fights: Number of fights to generate
        seed: Random seed

    Returns:
        DataFrame with synthetic fight data
    """
    logger.info(f"Creating synthetic dataset with {n_fights} fights...")

    np.random.seed(seed)

    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i*7) for i in range(n_fights)]

    # Generate fighter names
    fighters_pool = [f"Fighter_{i}" for i in range(200)]

    data = []

    for i, date in enumerate(dates):
        # Randomly select two fighters
        f1, f2 = np.random.choice(fighters_pool, size=2, replace=False)

        # Generate realistic features
        # Fighter 1 stats (career aggregates - SAFE)
        f1_wins = np.random.randint(0, 20)
        f1_losses = np.random.randint(0, 10)
        f1_slpm = np.random.uniform(2, 6)  # Strikes landed per minute
        f1_td_avg = np.random.uniform(0, 5)  # Takedowns per fight
        f1_str_acc = np.random.uniform(0.35, 0.55)  # Striking accuracy
        f1_td_acc = np.random.uniform(0.2, 0.6)  # Takedown accuracy

        # Fighter 2 stats (career aggregates - SAFE)
        f2_wins = np.random.randint(0, 20)
        f2_losses = np.random.randint(0, 10)
        f2_slpm = np.random.uniform(2, 6)
        f2_td_avg = np.random.uniform(0, 5)
        f2_str_acc = np.random.uniform(0.35, 0.55)
        f2_td_acc = np.random.uniform(0.2, 0.6)

        # Betting odds (pre-fight - SAFE)
        f1_odds = np.random.uniform(1.5, 3.0)
        f2_odds = np.random.uniform(1.5, 3.0)

        # Target (outcome)
        # Make it somewhat predictable based on features
        win_prob = (
            0.5 +  # Base
            0.1 * (f1_wins - f2_wins) / 20 +  # Win differential
            0.1 * (f1_slpm - f2_slpm) / 4 +  # Striking differential
            0.05 * (f1_str_acc - f2_str_acc) +  # Accuracy differential
            np.random.normal(0, 0.15)  # Noise
        )
        win_prob = np.clip(win_prob, 0.1, 0.9)

        target = 1 if np.random.random() < win_prob else 0  # 1 = f2 wins, 0 = f1 wins

        # Add some LEAKED features to test detection
        # These should be flagged by the audit
        if i % 10 == 0:  # Add leakage to 10% of data
            # LEAKED: Round 1 strikes (should be detected)
            f1_r1_strikes = np.random.randint(10, 50)
            f2_r1_strikes = np.random.randint(10, 50)

            # LEAKED: Total fight strikes (should be detected)
            f1_total_strikes = np.random.randint(50, 200)
            f2_total_strikes = np.random.randint(50, 200)
        else:
            f1_r1_strikes = np.nan
            f2_r1_strikes = np.nan
            f1_total_strikes = np.nan
            f2_total_strikes = np.nan

        data.append({
            'event_date': date,
            'f_1_name': f1,
            'f_2_name': f2,

            # SAFE features (career aggregates)
            'f_1_fighter_w': f1_wins,
            'f_1_fighter_l': f1_losses,
            'f_1_SlpM': f1_slpm,
            'f_1_TD_Avg': f1_td_avg,
            'f_1_Str_Acc': f1_str_acc,
            'f_1_TD_Acc': f1_td_acc,

            'f_2_fighter_w': f2_wins,
            'f_2_fighter_l': f2_losses,
            'f_2_SlpM': f2_slpm,
            'f_2_TD_Avg': f2_td_avg,
            'f_2_Str_Acc': f2_str_acc,
            'f_2_TD_Acc': f2_td_acc,

            # SAFE features (pre-fight odds)
            'f_1_odds': f1_odds,
            'f_2_odds': f2_odds,

            # LEAKED features (should be detected and flagged)
            'f_1_r1_strikes': f1_r1_strikes,
            'f_2_r1_strikes': f2_r1_strikes,
            'f_1_total_strikes_succ': f1_total_strikes,
            'f_2_total_strikes_succ': f2_total_strikes,

            # Target
            'target': target
        })

    df = pd.DataFrame(data)

    logger.success(f"✓ Created synthetic dataset:")
    logger.info(f"  {len(df)} fights")
    logger.info(f"  {len(df.columns)} features")
    logger.info(f"  Date range: {df['event_date'].min().date()} to {df['event_date'].max().date()}")
    logger.info(f"  Target distribution: {df['target'].mean():.1%} (class 1)")

    return df


def run_demo_audit():
    """Run audit on synthetic data"""
    logger.info("="*80)
    logger.info("LEAKAGE AUDIT DEMO - SYNTHETIC DATA")
    logger.info("="*80 + "\n")

    # Create synthetic data
    df = create_synthetic_ufc_data(n_fights=1000)

    # Save to temp file
    temp_file = Path('/tmp/synthetic_ufc_data.csv')
    df.to_csv(temp_file, index=False)
    logger.info(f"\n✓ Saved to: {temp_file}\n")

    # Run audit
    logger.info("="*80)
    logger.info("RUNNING LEAKAGE AUDIT")
    logger.info("="*80 + "\n")

    import subprocess
    result = subprocess.run([
        'python', 'scripts/audit_data_leakage.py',
        '--data', str(temp_file),
        '--target', 'target',
        '--date', 'event_date'
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    logger.info("\n" + "="*80)
    logger.info("DEMO COMPLETE")
    logger.info("="*80)

    logger.info("\nWhat to expect:")
    logger.info("1. ⚠️  Leaked features detected: f_1_r1_strikes, f_2_r1_strikes, etc.")
    logger.info("2. ✓  Safe features pass: f_1_fighter_w, f_1_SlpM, etc.")
    logger.info("3. ✓  Shuffle test: Random split should be 2-5% better than temporal")
    logger.info("4. ✓  No suspicious correlations (all < 0.5)")

    logger.info("\nTo run on your real dataset:")
    logger.info("python scripts/audit_data_leakage.py --data [path_to_golden_dataset.csv]")


if __name__ == "__main__":
    run_demo_audit()
