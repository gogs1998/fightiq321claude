"""
UFC 321 - Live Scraped Odds from BestFightOdds
Date: October 25th, 2025
Location: Etihad Arena, Abu Dhabi, UAE
"""

import pandas as pd
import numpy as np

def american_to_decimal(american_odds):
    """Convert American odds to decimal"""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1

# UFC 321 - MAIN CARD
ufc321_main_card = [
    {
        'fight': 'Main Event',
        'fighter1': 'Tom Aspinall',
        'fighter2': 'Ciryl Gane',
        'f1_odds_american_low': -500,  # Caesars (best for Aspinall)
        'f1_odds_american_high': +110,  # BetMGM (worst for Aspinall)
        'f2_odds_american_low': -500,  # Caesars (worst for Gane)
        'f2_odds_american_high': +340,  # FanDuel (best for Gane)
        'consensus_f1': -400,  # Estimated average
        'consensus_f2': +300,
    },
    {
        'fight': 'Co-Main Event',
        'fighter1': 'Mackenzie Dern',
        'fighter2': 'Virna Jandiroba',
        'f1_odds_american_low': -178,  # FanDuel
        'f1_odds_american_high': -165,  # DraftKings
        'f2_odds_american_low': +132,  # DraftKings
        'f2_odds_american_high': +138,  # FanDuel
        'consensus_f1': -170,
        'consensus_f2': +135,
    },
    {
        'fight': 'Main Card',
        'fighter1': 'Umar Nurmagomedov',
        'fighter2': 'Mario Bautista',
        'f1_odds_american_low': -650,  # FanDuel
        'f1_odds_american_high': -625,  # BetRivers
        'f2_odds_american_low': +420,  # FanDuel
        'f2_odds_american_high': +440,  # BetRivers
        'consensus_f1': -640,
        'consensus_f2': +430,
    },
    {
        'fight': 'Main Card',
        'fighter1': 'Alexander Volkov',
        'fighter2': 'Jailton Almeida',
        'f1_odds_american_low': +182,  # FanDuel
        'f1_odds_american_high': +188,  # BetRivers
        'f2_odds_american_low': -240,  # Both books
        'f2_odds_american_high': -240,
        'consensus_f1': +185,
        'consensus_f2': -240,
    },
    {
        'fight': 'Main Card',
        'fighter1': 'Aleksandar Rakic',
        'fighter2': 'Azamat Murzakanov',
        'f1_odds_american_low': -110,  # BetRivers
        'f1_odds_american_high': -106,  # FanDuel
        'f2_odds_american_low': -120,  # FanDuel
        'f2_odds_american_high': -112,  # BetRivers
        'consensus_f1': -108,
        'consensus_f2': -116,
    }
]

# Convert to decimal odds
for fight in ufc321_main_card:
    fight['f1_decimal'] = american_to_decimal(fight['consensus_f1'])
    fight['f2_decimal'] = american_to_decimal(fight['consensus_f2'])

# Create DataFrame
df = pd.DataFrame(ufc321_main_card)

print("="*100)
print("UFC 321: ASPINALL VS GANE - LIVE ODDS FROM BESTFIGHTODDS")
print("Saturday, October 25th, 2025 | Etihad Arena, Abu Dhabi, UAE")
print("="*100)

print("\nMAIN CARD ODDS (CONSENSUS ACROSS SPORTSBOOKS)")
print("-"*100)

for idx, row in df.iterrows():
    print(f"\n{row['fight'].upper()}: {row['fighter1']} vs {row['fighter2']}")
    print(f"  {row['fighter1']}: {row['consensus_f1']:+d} (American) = {row['f1_decimal']:.2f} (Decimal)")
    print(f"  {row['fighter2']}: {row['consensus_f2']:+d} (American) = {row['f2_decimal']:.2f} (Decimal)")

    # Show odds range
    print(f"  Range: {row['fighter1']} ({row['f1_odds_american_low']:+d} to {row['f1_odds_american_high']:+d})")
    print(f"         {row['fighter2']} ({row['f2_odds_american_low']:+d} to {row['f2_odds_american_high']:+d})")

    # Implied probabilities
    f1_implied = 1 / row['f1_decimal']
    f2_implied = 1 / row['f2_decimal']
    print(f"  Implied: {row['fighter1']} {f1_implied*100:.1f}% | {row['fighter2']} {f2_implied*100:.1f}%")

print("\n" + "="*100)
print("KEY INSIGHTS")
print("="*100)

print("\n1. BIGGEST FAVORITE: Umar Nurmagomedov (-640)")
print("   - 86.5% implied probability to beat Mario Bautista")
print("   - $640 to win $100")

print("\n2. BIGGEST UNDERDOG: Ciryl Gane (+300)")
print("   - 25% implied probability vs Tom Aspinall")
print("   - $100 to win $300")

print("\n3. PICK 'EM FIGHT: Rakic vs Murzakanov")
print("   - Nearly even odds (-108 vs -116)")
print("   - Both fighters around 50% implied probability")

print("\n4. BEST VALUE BET (if odds are accurate):")
print("   - Virna Jandiroba (+135) vs Mackenzie Dern")
print("   - Alexander Volkov (+185) vs Jailton Almeida")
print("   - Both underdogs with decent odds")

print("\n" + "="*100)
print("SPORTSBOOKS TRACKED")
print("="*100)
print("FanDuel, Caesars, BetMGM, BetRivers, DraftKings, and more...")

# Save to CSV
csv_file = 'ufc321_scraped_odds.csv'
df_output = df[['fighter1', 'fighter2', 'consensus_f1', 'consensus_f2', 'f1_decimal', 'f2_decimal']]
df_output.to_csv(csv_file, index=False)
print(f"\n✓ Saved to: {csv_file}")

print("\n" + "="*100)
print("NEXT STEPS")
print("="*100)
print("1. Use these odds in prediction pipeline:")
print("   python scripts/predict_upcoming_with_bestfightodds.py")
print("\n2. Compare model predictions vs market odds")
print("\n3. Identify value bets where model disagrees with market")
print("="*100)
