# Repository Contents

## ✅ What's Included (Ready to Use)

### Production Models (~9.4MB)
- `models/xgboost_production.json` - XGBoost model (1.6MB)
- `models/lightgbm_production.txt` - LightGBM model (1.8MB)
- `models/ensemble_production.pkl` - Ensemble wrapper (47KB)
- **Training**: 7,317 fights (1994-2024)
- **Test Accuracy**: 70.8% on 2025 holdout
- **Backtested ROI**: +146.9%

### Alternative Models (Experiments)
- `models/xgboost_baseline.json` - No-odds baseline
- `models/xgboost_with_odds.json` - With odds features
- `models/xgboost_advanced.json` - Advanced feature engineering
- *(Same for LightGBM and ensemble versions)*

### Predictions
- `predictions_ufc321.csv` - **UFC 321 predictions** (26 fights, Oct 25, 2025)
- `predictions_production.csv` - Production model validation results
- `predictions_with_odds_model.csv` - Historical predictions
- `UFC321_PREDICTIONS_FULL.txt` - Complete UFC 321 breakdown

### Scripts
- `scripts/predict_upcoming_ufc321.py` - **Main prediction pipeline** ⭐
- `scripts/train_production.py` - Retrain production models
- `scripts/train_baseline.py` - Train baseline (no odds)
- `scripts/train_with_odds.py` - Train with odds features
- `scripts/backtest_yearly_holdouts.py` - Year-by-year validation
- `scripts/backtest_actual_roi_fixed.py` - ROI backtesting

### Source Code
- `src/data/loaders.py` - Data loading with leak detection
- `src/data/preprocessing.py` - Feature engineering
- `src/data/splitters.py` - Temporal train/test splits
- `src/models/ensemble.py` - Ensemble model wrapper
- `src/utils/config.py` - Configuration management

### Configuration
- `config/config.yaml` - Central configuration file
- `requirements.txt` - Python dependencies

### Documentation
- `README.md` - Main documentation
- `SETUP.md` - Setup instructions
- `ufc321_full_breakdown.py` - UFC 321 analysis script

## ⚠️ NOT Included (Need to Download)

### UFC Historical Dataset (379MB)
**Why not included**: Too large for GitHub (100MB limit)

**Where to get it**:
1. **Option A**: Clone FightIQ repository
   ```bash
   git clone https://github.com/bfortuner/fightiq.git
   ```
   Dataset at: `FightIQ/data/UFC_full_data_golden.csv`

2. **Option B**: Scrape fresh data (takes several hours)

**Required for**:
- Retraining models
- Running backtests
- Generating new predictions for fighters not in cache

**NOT required for**:
- Using pre-trained models
- Making predictions for fighters in database (up to Oct 2025)

### Odds API Key (Free)
**Why not included**: Secret credential

**Where to get it**: https://the-odds-api.com/
- Free tier: 500 calls/month
- Each prediction run: 1-2 calls

**Required for**:
- Fetching upcoming fights
- Getting real-time odds

**NOT required for**:
- Training models
- Running backtests on historical data

## 🚀 What You Can Do Right Now

### With Just This Repository:

1. **View UFC 321 Predictions** ✅
   - Open `UFC321_PREDICTIONS_FULL.txt`
   - See 18 high-confidence betting recommendations

2. **Inspect Models** ✅
   - Load models in Python
   - Analyze feature importance
   - Understand model architecture

3. **Read Code & Documentation** ✅
   - Study prediction pipeline
   - Learn feature engineering techniques
   - Understand betting strategy

### With Odds API Key:

4. **Predict Upcoming Fights** ✅
   ```bash
   python scripts/predict_upcoming_ufc321.py
   ```
   - Fetches real-time odds
   - Generates predictions
   - Recommends bets

### With UFC Dataset:

5. **Retrain Models** ✅
   ```bash
   python scripts/train_production.py
   ```

6. **Run Backtests** ✅
   ```bash
   python scripts/backtest_yearly_holdouts.py
   ```

7. **Experiment with Features** ✅
   - Modify `src/data/preprocessing.py`
   - Train custom models
   - Test new strategies

## 📊 File Sizes

| Category | Total Size | Included |
|----------|-----------|----------|
| Models | 9.4 MB | ✅ Yes |
| Predictions | 217 KB | ✅ Yes |
| Code & Config | 142 KB | ✅ Yes |
| **Dataset** | **379 MB** | ❌ No (too large) |

**Total Repository Size**: ~10 MB (well within GitHub limits)

## 🔄 Updating

### Get Latest Predictions

```bash
git pull origin main
```

### Update Models with New Data

1. Get latest UFC dataset
2. Run: `python scripts/train_production.py`
3. Commit: `git add models/ && git commit -m "Update models"`

### Contribute Improvements

1. Fork repository
2. Make changes
3. Submit pull request

## 📝 Notes

- **Models valid until**: Oct 2025 (dataset cutoff)
- **Retrain recommended**: After major UFC events
- **API calls**: Monitor usage to stay within free tier
- **Dataset source**: FightIQ project (open source)

---

**Summary**: Repository includes everything needed to **make predictions immediately**, but requires dataset download for retraining or experimenting with new features.
