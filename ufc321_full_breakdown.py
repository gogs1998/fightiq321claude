import pandas as pd

df = pd.read_csv('predictions_ufc321.csv')

# Known Vegas odds from web search
vegas_odds = {
    'Tom Aspinall': {'vs': 'Ciryl Gane', 'our_odds': 1.25, 'vegas_odds': 1.26, 'vegas_opponent': 4.30},
    'Virna Jandiroba': {'vs': 'Mackenzie Dern', 'our_odds': 2.40, 'vegas_odds': 2.45, 'vegas_opponent': 1.57},
    'Umar Nurmagomedov': {'vs': 'Mario Bautista', 'our_odds': 1.17, 'vegas_odds': 1.17, 'vegas_opponent': 5.25},
    'Alexander Volkov': {'vs': 'Jailton Almeida', 'our_odds': 2.81, 'vegas_odds': 2.85, 'vegas_opponent': 1.44},
    'Aleksandar Rakic': {'vs': 'Azamat Murzakanov', 'our_odds': 1.95, 'vegas_odds': 1.95, 'vegas_opponent': 1.87},
    'Nathaniel Wood': {'vs': 'Jose Delgado', 'our_odds': 2.24, 'vegas_odds': 2.30, 'vegas_opponent': 1.62},
}

print('='*120)
print('UFC 321: ASPINALL vs GANE - COMPLETE CARD BREAKDOWN')
print('Saturday, October 25, 2025 | Etihad Arena, Abu Dhabi, UAE')
print('='*120)

print('\n' + '='*120)
print('MAIN CARD (PPV)')
print('='*120)

main_card = [
    'Tom Aspinall',
    'Virna Jandiroba',
    'Umar Nurmagomedov',
    'Alexander Volkov',
    'Aleksandar Rakic'
]

for fighter in main_card:
    row = df[(df['fighter1'].str.contains(fighter, na=False)) | (df['fighter2'].str.contains(fighter, na=False))]

    if len(row) == 0:
        continue

    row = row.iloc[0]

    f1 = row['fighter1']
    f2 = row['fighter2']

    # Determine which is our pick
    winner = row['predicted_winner']
    confidence = row['confidence'] * 100

    f1_prob = row['prob_f1_wins'] * 100
    f2_prob = row['prob_f2_wins'] * 100

    our_f1_odds = row['fighter1_odds']
    our_f2_odds = row['fighter2_odds']

    rec = row['recommended_bet']

    # Check if we have Vegas odds
    has_vegas = fighter in vegas_odds

    print(f'\n{f1} vs {f2}')
    print('-' * 120)

    if has_vegas:
        vdata = vegas_odds[fighter]
        if f1 == fighter:
            print(f'  ODDS: {f1}: {our_f1_odds:.2f} (Vegas: {vdata["vegas_odds"]:.2f}) | {f2}: {our_f2_odds:.2f} (Vegas: {vdata["vegas_opponent"]:.2f})')
        else:
            print(f'  ODDS: {f1}: {our_f1_odds:.2f} (Vegas: {vdata["vegas_opponent"]:.2f}) | {f2}: {our_f2_odds:.2f} (Vegas: {vdata["vegas_odds"]:.2f})')
    else:
        print(f'  ODDS: {f1}: {our_f1_odds:.2f} | {f2}: {our_f2_odds:.2f}')

    print(f'  MODEL: {f1}: {f1_prob:.1f}% | {f2}: {f2_prob:.1f}%')
    print(f'  PICK: {winner} ({confidence:.1f}% confidence)')

    # Determine favorite/underdog
    if our_f1_odds < our_f2_odds:
        favorite = f1
        underdog = f2
        fav_prob = f1_prob
        dog_prob = f2_prob
        fav_odds = our_f1_odds
        dog_odds = our_f2_odds
    else:
        favorite = f2
        underdog = f1
        fav_prob = f2_prob
        dog_prob = f1_prob
        fav_odds = our_f2_odds
        dog_odds = our_f1_odds

    print(f'  MARKET: {favorite} (favorite at {fav_odds:.2f}) vs {underdog} (underdog at {dog_odds:.2f})')

    # Analysis
    if winner == favorite:
        market_prob = 1/fav_odds * 100
        edge = confidence - market_prob
        print(f'  ANALYSIS: Backing the favorite | Edge: {edge:+.1f}%')
    else:
        market_prob = 1/dog_odds * 100
        edge = confidence - market_prob
        print(f'  ANALYSIS: UNDERDOG PICK! | Edge: {edge:+.1f}%')

    # Recommendation
    if rec.startswith('BET'):
        exp_roi = (confidence/100 * (fav_odds if winner == favorite else dog_odds) - 1) * 100
        print(f'  RECOMMENDATION: {rec}')
        print(f'  Expected ROI: {exp_roi:+.1f}%')
    else:
        print(f'  RECOMMENDATION: {rec}')
        if confidence < 60:
            print(f'  REASON: Confidence ({confidence:.1f}%) below 60% threshold')


print('\n\n' + '='*120)
print('PRELIMINARY CARD (ESPN+/FX)')
print('='*120)

prelims = df[~df['fighter1'].isin([r['fighter1'] for i, r in df.iterrows() if any(mc in r['fighter1'] or mc in r['fighter2'] for mc in main_card)])].copy()

# Sort by confidence
prelims = prelims.sort_values('confidence', ascending=False)

for idx, row in prelims.iterrows():
    f1 = row['fighter1']
    f2 = row['fighter2']
    winner = row['predicted_winner']
    confidence = row['confidence'] * 100

    f1_prob = row['prob_f1_wins'] * 100
    f2_prob = row['prob_f2_wins'] * 100

    our_f1_odds = row['fighter1_odds']
    our_f2_odds = row['fighter2_odds']

    rec = row['recommended_bet']

    print(f'\n{f1} vs {f2}')
    print('-' * 120)
    print(f'  ODDS: {f1}: {our_f1_odds:.2f} | {f2}: {our_f2_odds:.2f}')
    print(f'  MODEL: {f1}: {f1_prob:.1f}% | {f2}: {f2_prob:.1f}%')
    print(f'  PICK: {winner} ({confidence:.1f}% confidence)')

    # Determine favorite/underdog
    if our_f1_odds < our_f2_odds:
        favorite = f1
        underdog = f2
        dog_odds = our_f2_odds
        fav_odds = our_f1_odds
    else:
        favorite = f2
        underdog = f1
        dog_odds = our_f1_odds
        fav_odds = our_f2_odds

    if winner == underdog:
        market_prob = 1/dog_odds * 100
        edge = confidence - market_prob
        print(f'  ANALYSIS: UNDERDOG PICK! {underdog} at {dog_odds:.2f} | Edge: {edge:+.1f}%')
    else:
        market_prob = 1/fav_odds * 100
        edge = confidence - market_prob
        print(f'  ANALYSIS: Favorite pick | Edge: {edge:+.1f}%')

    if rec.startswith('BET'):
        pick_odds = our_f1_odds if winner == f1 else our_f2_odds
        exp_roi = (confidence/100 * pick_odds - 1) * 100
        print(f'  RECOMMENDATION: BET {winner} | Expected ROI: {exp_roi:+.1f}%')
    else:
        print(f'  RECOMMENDATION: {rec}')


print('\n\n' + '='*120)
print('BETTING SUMMARY')
print('='*120)

bets = df[df['recommended_bet'].str.startswith('BET', na=False)]
passes = df[df['recommended_bet'] == 'PASS']
no_data = df[df['recommended_bet'].str.contains('Unable', na=False)]

print(f'\nTotal fights: {len(df)}')
print(f'High-confidence bets: {len(bets)} ({len(bets)/len(df)*100:.1f}%)')
print(f'Pass (low confidence/edge): {len(passes)} ({len(passes)/len(df)*100:.1f}%)')
print(f'No data (fighters not found): {len(no_data)} ({len(no_data)/len(df)*100:.1f}%)')

print('\n' + '='*120)
print('TOP 5 UNDERDOG VALUE PLAYS')
print('='*120)

underdog_picks = []
for idx, row in bets.iterrows():
    f1_odds = row['fighter1_odds']
    f2_odds = row['fighter2_odds']
    winner = row['predicted_winner']

    if winner == row['fighter1'] and f1_odds > f2_odds:
        underdog_picks.append(row)
    elif winner == row['fighter2'] and f2_odds > f1_odds:
        underdog_picks.append(row)

underdog_df = pd.DataFrame(underdog_picks)
if len(underdog_df) > 0:
    underdog_df['pick_odds'] = underdog_df.apply(
        lambda r: r['fighter1_odds'] if r['predicted_winner'] == r['fighter1'] else r['fighter2_odds'],
        axis=1
    )
    underdog_df['exp_roi'] = underdog_df.apply(
        lambda r: (r['confidence'] * r['pick_odds'] - 1) * 100,
        axis=1
    )
    underdog_df = underdog_df.sort_values('exp_roi', ascending=False).head(5)

    for idx, row in underdog_df.iterrows():
        loser = row['fighter2'] if row['predicted_winner'] == row['fighter1'] else row['fighter1']
        print(f"{row['predicted_winner']} over {loser}")
        print(f"  Confidence: {row['confidence']*100:.1f}% | Odds: {row['pick_odds']:.2f} | Expected ROI: {row['exp_roi']:+.1f}%")


print('\n' + '='*120)
print('TOP 5 HIGHEST CONFIDENCE PICKS')
print('='*120)

top_conf = bets.sort_values('confidence', ascending=False).head(5)
for idx, row in top_conf.iterrows():
    loser = row['fighter2'] if row['predicted_winner'] == row['fighter1'] else row['fighter1']
    pick_odds = row['fighter1_odds'] if row['predicted_winner'] == row['fighter1'] else row['fighter2_odds']
    print(f"{row['predicted_winner']} over {loser}")
    print(f"  Confidence: {row['confidence']*100:.1f}% | Odds: {pick_odds:.2f}")

print('\n' + '='*120)
print('MODEL PERFORMANCE')
print('='*120)
print('Training: 1994-2024 (7,317 fights)')
print('Test Accuracy: 70.8% (2025 holdout)')
print('Backtested ROI: +146.9% (Conservative strategy)')
print('Strategy: 60% confidence threshold, fixed stakes')
print('='*120)
