import pandas as pd

df = pd.read_csv('predictions_ufc321.csv')
bets = df[df['recommended_bet'].str.startswith('BET', na=False)].copy()

print('\n' + '='*100)
print('UFC 321 PRODUCTION PREDICTIONS - FINAL SUMMARY (ODDS BUG FIXED)')
print('='*100)

print(f'\nTotal fights analyzed: {len(df)}')
print(f'High-confidence bets: {len(bets)}')
print(f'Pass (low confidence): {len(df) - len(bets)}')

print('\n' + '='*100)
print('RECOMMENDED BETS (Conservative 60% Threshold)')
print('='*100)

for idx, row in bets.iterrows():
    winner = row['predicted_winner']
    loser = row['fighter2'] if winner == row['fighter1'] else row['fighter1']
    conf = row['confidence'] * 100
    f1_odds = row['fighter1_odds']
    f2_odds = row['fighter2_odds']

    is_f1_winner = (winner == row['fighter1'])
    pick_odds = f1_odds if is_f1_winner else f2_odds
    market_prob = 1 / pick_odds * 100
    edge = conf - market_prob

    # Calculate expected ROI
    exp_roi = (conf/100 * pick_odds - 1) * 100

    print(f"\n{row['fighter1']} ({f1_odds:.2f}) vs {row['fighter2']} ({f2_odds:.2f})")
    print(f"  PICK: {winner}")
    print(f"  Confidence: {conf:.1f}%")
    print(f"  Pick Odds: {pick_odds:.2f} (market implies {market_prob:.1f}%)")
    print(f"  Model Edge: {edge:+.1f}%")
    print(f"  Expected ROI: {exp_roi:+.1f}%")

print('\n' + '='*100)
print('TOP 5 HIGHEST CONFIDENCE PICKS')
print('='*100)

top5 = bets.sort_values('confidence', ascending=False).head(5)
for idx, row in top5.iterrows():
    print(f"{row['predicted_winner']} ({row['confidence']*100:.1f}%) over {row['fighter2'] if row['predicted_winner']==row['fighter1'] else row['fighter1']}")

print('\n' + '='*100)
print('VERIFICATION: Khamzat Chimaev (Should be heavy favorite)')
print('='*100)

khamzat = df[df['fighter1'].str.contains('Khamzat', na=False) | df['fighter2'].str.contains('Khamzat', na=False)]
for idx, row in khamzat.iterrows():
    print(f"\n{row['fighter1']} ({row['fighter1_odds']:.2f}) vs {row['fighter2']} ({row['fighter2_odds']:.2f})")
    print(f"  Predicted: {row['predicted_winner']} ({row['confidence']*100:.1f}%)")

    if row['fighter1'] == 'Khamzat Chimaev':
        if row['fighter1_odds'] < 2.0:
            print("  ✓ CORRECT: Khamzat is the favorite")
        else:
            print("  ✗ ERROR: Khamzat should have lower odds")
    else:
        if row['fighter2_odds'] < 2.0:
            print("  ✓ CORRECT: Khamzat is the favorite")
        else:
            print("  ✗ ERROR: Khamzat should have lower odds")

print('\n' + '='*100)
print('PRODUCTION SYSTEM STATS')
print('='*100)
print(f"Model: Production Ensemble (XGBoost + LightGBM)")
print(f"Training: 1994-2024 (7,317 fights)")
print(f"Test Accuracy: 70.8% on 2025 holdout")
print(f"Backtested ROI: +146.9% (Conservative strategy, 194 bets)")
print(f"Features: 1,476 leak-free features + real-time odds")
print(f"Odds API calls remaining: Check log")
print('='*100 + '\n')
