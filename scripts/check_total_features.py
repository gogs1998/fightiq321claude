"""
Check if _total features in FightIQ are historical or current-fight
"""
import pandas as pd
from pathlib import Path

# Load FightIQ dataset
df_fightiq = pd.read_csv(Path('D:/Codex/FightIQ/data/UFC_full_data_golden.csv'))

print(f"FightIQ Total columns: {len(df_fightiq.columns)}")
print(f"FightIQ Total fights: {len(df_fightiq)}")

# Check if _total features exist
total_features = [col for col in df_fightiq.columns if '_total' in col]
print(f"\n_total features in FightIQ: {len(total_features)}")
print("\nSample _total features:")
for col in total_features[:20]:
    print(f"  {col}")

# Check specific feature
if 'f_1_head_succ_total' in df_fightiq.columns:
    print(f"\nf_1_head_succ_total EXISTS in FightIQ:")
    print(f"  Mean: {df_fightiq['f_1_head_succ_total'].mean():.2f}")
    print(f"  Max: {df_fightiq['f_1_head_succ_total'].max():.0f}")
    print(f"  Sample: {df_fightiq['f_1_head_succ_total'].head(10).tolist()}")
else:
    print("\nf_1_head_succ_total NOT FOUND in FightIQ dataset")

# Load our dataset
df_ours = pd.read_csv(Path('D:/Codex/UFC-Master-Pipeline/data/UFC_full_data_golden.csv'))

print(f"\n\nOur dataset Total columns: {len(df_ours.columns)}")
print(f"Our dataset Total fights: {len(df_ours)}")

# Compare
total_features_ours = [col for col in df_ours.columns if '_total' in col]
print(f"\n_total features in our dataset: {len(total_features_ours)}")

if 'f_1_head_succ_total' in df_ours.columns:
    print(f"\nf_1_head_succ_total in OUR dataset:")
    print(f"  Mean: {df_ours['f_1_head_succ_total'].mean():.2f}")
    print(f"  Max: {df_ours['f_1_head_succ_total'].max():.0f}")
    print(f"  Sample: {df_ours['f_1_head_succ_total'].head(10).tolist()}")
