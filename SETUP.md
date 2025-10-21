# Setup Instructions

## Quick Start (Using Pre-Trained Models)

The repository includes **pre-trained production models**, so you can start making predictions immediately!

```bash
git clone https://github.com/gogs1998/fightiq321claude.git
cd fightiq321claude
pip install -r requirements.txt
```

### Make Predictions for Upcoming Fights

```bash
python scripts/predict_upcoming_ufc321.py
```

**Note:** You'll need an Odds API key. Get one free at https://the-odds-api.com/

Edit `scripts/predict_upcoming_ufc321.py` and set:
```python
ODDS_API_KEY = "your_api_key_here"
```

## Full Setup (For Retraining Models)

If you want to retrain models with the latest data, you'll need the UFC historical dataset.

### Option 1: Use FightIQ Dataset (Recommended)

1. Clone FightIQ repository:
```bash
cd D:/Codex  # or your preferred directory
git clone https://github.com/bfortuner/fightiq.git FightIQ
```

2. The dataset should be at: `D:/Codex/FightIQ/data/UFC_full_data_golden.csv`

3. Update paths in `config/config.yaml` if needed:
```yaml
paths:
  golden_dataset: "D:/Codex/FightIQ/data/UFC_full_data_golden.csv"
```

### Option 2: Download Dataset Directly

If FightIQ dataset is not available, you can scrape fresh data:

1. Install scraping dependencies:
```bash
pip install beautifulsoup4 selenium
```

2. Run data collection (this will take several hours):
```bash
python scripts/scrape_ufc_data.py  # Not included - use FightIQ's scrapers
```

### Dataset Requirements

The dataset should have these columns:
- `event_date`: Fight date
- `f_1_name`, `f_2_name`: Fighter names
- `winner_encoded`: 0 (F1 wins), 1 (F2 wins)
- Historical features: Career stats, rolling averages, etc.
- Odds features: `f_1_odds`, `f_2_odds`, etc.

**Total expected**: ~8,000 fights, ~5,400 columns

## Retrain Production Models

Once you have the dataset:

```bash
python scripts/train_production.py
```

This will:
- Train on 1994-2024 data
- Validate on 2025 holdout
- Save new models to `models/`
- Expected runtime: 5-10 minutes

## Directory Structure

```
fightiq321claude/
├── config/
│   └── config.yaml              # Update paths here
├── models/                       # ✅ Pre-trained models included
│   ├── xgboost_production.json
│   ├── lightgbm_production.txt
│   └── ensemble_production.pkl
├── scripts/
│   ├── predict_upcoming_ufc321.py   # ✅ Ready to use
│   ├── train_production.py          # Requires dataset
│   └── backtest_*.py                # Requires dataset
└── predictions_ufc321.csv       # ✅ UFC 321 predictions included
```

## What's Included vs What You Need

### ✅ Included (Ready to Use)
- Pre-trained production models (XGBoost, LightGBM, Ensemble)
- UFC 321 predictions
- All scripts and source code
- Configuration files

### ⚠️ Required for Retraining
- UFC historical dataset (379MB) - Get from FightIQ
- Odds API key (free) - Get from the-odds-api.com

## Troubleshooting

### "golden_dataset not found"
Update path in `config/config.yaml`:
```yaml
paths:
  golden_dataset: "YOUR_PATH/UFC_full_data_golden.csv"
```

### "Module not found"
Install dependencies:
```bash
pip install -r requirements.txt
```

### "No upcoming fights found"
Check your Odds API key and remaining API calls:
- Free tier: 500 calls/month
- Each prediction run uses 1-2 calls

### Models not loading
The pre-trained models are in `models/` directory. If missing:
```bash
git lfs pull  # If using Git LFS
# or
python scripts/train_production.py  # Retrain from scratch
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/gogs1998/fightiq321claude/issues
- FightIQ Documentation: https://github.com/bfortuner/fightiq

## License

MIT License - See LICENSE file
