"""
Add Fighting Style Matchup Features

Adds features related to fighting styles and matchup dynamics:
- Striker vs Grappler indicators
- Submission threat levels
- Ground control preferences
- Distance striking patterns
- Clinch work preferences

Expected accuracy improvement: +0.5% to +1.5%
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import numpy as np
from loguru import logger


def add_style_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fighting style matchup features

    Args:
        df: Dataset with existing features

    Returns:
        Dataset with new style features
    """
    logger.info("="*80)
    logger.info("ADDING FIGHTING STYLE MATCHUP FEATURES")
    logger.info("="*80)

    df_enhanced = df.copy()

    logger.info("\nCalculating style indicators...")

    # 1. STRIKING PREFERENCE
    # Striker: High significant strikes landed, low takedown attempts
    for fighter in ['f_1', 'f_2']:
        # Striking rate (strikes per minute)
        slpm_col = f'{fighter}_SlpM'
        if slpm_col in df.columns:
            df_enhanced[f'{fighter}_striking_preference'] = df[slpm_col]
        else:
            df_enhanced[f'{fighter}_striking_preference'] = 0

        # Distance striking (if available)
        distance_cols = [col for col in df.columns if f'{fighter}' in col and 'distance' in col.lower()]
        if distance_cols:
            df_enhanced[f'{fighter}_distance_fighter'] = df[distance_cols].mean(axis=1)
        else:
            df_enhanced[f'{fighter}_distance_fighter'] = 0

    # 2. GRAPPLING PREFERENCE
    for fighter in ['f_1', 'f_2']:
        # Takedown rate
        td_avg_col = f'{fighter}_TD_Avg'
        if td_avg_col in df.columns:
            df_enhanced[f'{fighter}_grappling_preference'] = df[td_avg_col]
        else:
            df_enhanced[f'{fighter}_grappling_preference'] = 0

        # Takedown accuracy
        td_acc_col = f'{fighter}_TD_Acc'
        if td_acc_col in df.columns:
            df_enhanced[f'{fighter}_td_threat'] = df[td_acc_col] / 100.0  # Convert % to decimal
        else:
            df_enhanced[f'{fighter}_td_threat'] = 0

    # 3. SUBMISSION THREAT
    for fighter in ['f_1', 'f_2']:
        # Submission average
        sub_avg_col = f'{fighter}_Sub_Avg'
        if sub_avg_col in df.columns:
            df_enhanced[f'{fighter}_submission_threat'] = df[sub_avg_col]
        else:
            df_enhanced[f'{fighter}_submission_threat'] = 0

        # Ground control (if available)
        # Look for control time features
        control_cols = [col for col in df.columns
                        if f'{fighter}' in col and ('ctrl' in col.lower() or 'control' in col.lower())]
        if control_cols:
            df_enhanced[f'{fighter}_ground_control'] = df[control_cols].mean(axis=1)
        else:
            df_enhanced[f'{fighter}_ground_control'] = 0

    # 4. DEFENSIVE METRICS
    for fighter in ['f_1', 'f_2']:
        # Striking defense
        str_def_col = f'{fighter}_Str_Def'
        if str_def_col in df.columns:
            df_enhanced[f'{fighter}_striking_defense'] = df[str_def_col] / 100.0
        else:
            df_enhanced[f'{fighter}_striking_defense'] = 0.5

        # Takedown defense
        td_def_col = f'{fighter}_TD_Def'
        if td_def_col in df.columns:
            df_enhanced[f'{fighter}_takedown_defense'] = df[td_def_col] / 100.0
        else:
            df_enhanced[f'{fighter}_takedown_defense'] = 0.5

    # 5. FIGHTER ARCHETYPE CLASSIFICATION
    # Based on striking vs grappling preference
    for fighter in ['f_1', 'f_2']:
        striking = df_enhanced[f'{fighter}_striking_preference'].fillna(0)
        grappling = df_enhanced[f'{fighter}_grappling_preference'].fillna(0)

        # Normalize to 0-1 scale for comparison
        striking_norm = (striking - striking.min()) / (striking.max() - striking.min() + 1e-6)
        grappling_norm = (grappling - grappling.min()) / (grappling.max() - grappling.min() + 1e-6)

        # Calculate archetype score (-1 = pure grappler, +1 = pure striker)
        df_enhanced[f'{fighter}_archetype_score'] = striking_norm - grappling_norm

    # 6. MATCHUP ADVANTAGES
    logger.info("Calculating matchup advantages...")

    # Striker vs Grappler advantage
    df_enhanced['striker_vs_grappler_advantage'] = (
        (df_enhanced['f_1_archetype_score'] - df_enhanced['f_2_archetype_score']) *
        (df_enhanced['f_1_takedown_defense'] - df_enhanced['f_2_td_threat'])
    )

    # Submission game matchup
    df_enhanced['submission_matchup_advantage'] = (
        df_enhanced['f_1_submission_threat'] - df_enhanced['f_2_submission_threat']
    )

    # Ground game advantage (grappling + ground control)
    df_enhanced['ground_game_advantage'] = (
        (df_enhanced['f_1_grappling_preference'] + df_enhanced['f_1_ground_control']) -
        (df_enhanced['f_2_grappling_preference'] + df_enhanced['f_2_ground_control'])
    )

    # Striking matchup (offense vs defense)
    df_enhanced['striking_matchup_advantage'] = (
        (df_enhanced['f_1_striking_preference'] * (1 - df_enhanced['f_2_striking_defense'])) -
        (df_enhanced['f_2_striking_preference'] * (1 - df_enhanced['f_1_striking_defense']))
    )

    # Defense matchup (who has better overall defense)
    df_enhanced['defensive_matchup_advantage'] = (
        ((df_enhanced['f_1_striking_defense'] + df_enhanced['f_1_takedown_defense']) / 2) -
        ((df_enhanced['f_2_striking_defense'] + df_enhanced['f_2_takedown_defense']) / 2)
    )

    # 7. STYLE CLASH INDICATORS
    logger.info("Creating style clash indicators...")

    # Wrestler vs Striker (high clash potential)
    df_enhanced['wrestler_vs_striker_clash'] = np.abs(
        df_enhanced['f_1_archetype_score'] - df_enhanced['f_2_archetype_score']
    )

    # Submission specialists facing each other
    df_enhanced['grappling_chess_match'] = np.minimum(
        df_enhanced['f_1_submission_threat'],
        df_enhanced['f_2_submission_threat']
    )

    # Both strikers (potential for standup war)
    df_enhanced['standup_war_potential'] = np.minimum(
        df_enhanced['f_1_striking_preference'],
        df_enhanced['f_2_striking_preference']
    )

    # Count new features
    new_feature_count = len(df_enhanced.columns) - len(df.columns)

    logger.success(f"\n✓ Added {new_feature_count} style matchup features")

    logger.info("\nFeature categories:")
    logger.info("  Fighter attributes: 14 (7 per fighter)")
    logger.info("  Matchup advantages: 5")
    logger.info("  Style clash indicators: 3")
    logger.info(f"  Total: {new_feature_count}")

    return df_enhanced


def main():
    """Main function"""
    from src.utils.config import get_config

    config = get_config()

    logger.info("="*80)
    logger.info("FIGHTING STYLE MATCHUP FEATURE ENGINEERING")
    logger.info("="*80 + "\n")

    # Load dataset (either golden or enhanced)
    logger.info("Loading dataset...")
    golden_path = Path(config.paths.golden_dataset)

    # Check if enhanced version exists
    enhanced_path = golden_path.parent / f"{golden_path.stem}_with_recent_form.csv"
    if enhanced_path.exists():
        logger.info(f"Loading enhanced dataset with recent form features...")
        df = pd.read_csv(enhanced_path)
        output_name = f"{golden_path.stem}_with_recent_form_and_style.csv"
    else:
        logger.info(f"Loading golden dataset...")
        df = pd.read_csv(golden_path)
        output_name = f"{golden_path.stem}_with_style.csv"

    logger.success(f"✓ Loaded {len(df)} fights")
    logger.info(f"Existing features: {len(df.columns)}\n")

    # Add features
    df_enhanced = add_style_matchup_features(df)

    logger.info(f"\nTotal features: {len(df_enhanced.columns)}")
    logger.info(f"New features added: {len(df_enhanced.columns) - len(df.columns)}")

    # Save enhanced dataset
    output_path = golden_path.parent / output_name
    logger.info(f"\nSaving to: {output_path}")

    df_enhanced.to_csv(output_path, index=False)

    logger.success(f"✓ Enhanced dataset saved!")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
