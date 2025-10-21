# UFC Master Pipeline - Production Fight Prediction System

![UFC Prediction System](https://img.shields.io/badge/Accuracy-70.8%25-brightgreen)
![ROI](https://img.shields.io/badge/Backtested_ROI-+146.9%25-success)
![Python](https://img.shields.io/badge/Python-3.10+-blue)

**World-class UFC fight prediction system** achieving 70.8% accuracy on 2025 holdout data and +146.9% backtested ROI.

Built with Claude Code - combines historical fight statistics with real-time betting odds to identify high-value betting opportunities.

## 🎯 Key Features

- **Production-Ready Predictions**: Full pipeline from Odds API to betting recommendations
- **1,476 Leak-Free Features**: Rigorous data validation prevents future information leakage
- **Ensemble Models**: XGBoost + LightGBM with optimized hyperparameters
- **Real-Time Integration**: Fetches upcoming fights and odds from The Odds API
- **Conservative Strategy**: 60% confidence threshold, proven profitable on historical data

## 📊 Performance

| Metric | Value |
|--------|-------|
| Test Accuracy (2025) | 70.8% |
| Test AUC | 0.7292 |
| Backtested ROI | +146.9% |
| Win Rate | 75.8% |
| Training Data | 7,317 fights (1994-2024) |

## 🚀 Quick Start

### Option 1: Kaggle Notebook (Recommended for Beginners)

**Run the complete pipeline in your browser - no setup required!**

1. Go to Kaggle: [UFC Master Pipeline Notebook](https://www.kaggle.com/)
2. Upload `UFC_Master_Pipeline_Production_System.ipynb`
3. Add the FightIQ UFC dataset
4. Click "Run All"

See [KAGGLE_NOTEBOOK_GUIDE.md](KAGGLE_NOTEBOOK_GUIDE.md) for detailed instructions.

### Option 2: Local Installation

```bash
pip install -r requirements.txt
```

### Predict Upcoming UFC Event

```python
python scripts/predict_upcoming_ufc321.py
```

This will:
1. Fetch upcoming fights from Odds API
2. Match fighters to historical database
3. Generate predictions using production ensemble
4. Output betting recommendations with expected ROI

## 📁 Project Structure

```
UFC-Master-Pipeline/
├── UFC_Master_Pipeline_Production_System.ipynb  # 📓 Kaggle notebook
├── KAGGLE_NOTEBOOK_GUIDE.md                    # 📖 Notebook usage guide
├── config/
│   └── config.yaml              # Central configuration
├── scripts/
│   ├── train_production.py      # Train production model (1994-2024)
│   ├── predict_upcoming_ufc321.py   # Predict upcoming fights
│   ├── backtest_yearly_holdouts.py  # Year-by-year validation
│   └── backtest_actual_roi_fixed.py # ROI backtesting
├── src/
│   ├── data/
│   │   └── loaders.py           # Data loading with leak detection
│   └── utils/
│       └── config.py            # Configuration management
└── models/
    ├── xgboost_production.json  # Pre-trained XGBoost (1.6MB)
    ├── lightgbm_production.txt  # Pre-trained LightGBM (1.8MB)
    └── ensemble_production.pkl  # Ensemble wrapper (47KB)
```

## 🎲 UFC 321 Predictions (Oct 25, 2025)

### Top Value Picks

| Fighter | Opponent | Confidence | Odds | Expected ROI |
|---------|----------|------------|------|--------------|
| **Virna Jandiroba** | Mackenzie Dern | 84.4% | 2.45 | +102.2% |
| **Jack Della Maddalena** | Islam Makhachev | 76.8% | 3.14 | +141.2% |
| **Belal Muhammad** | Ian Garry | 84.2% | 2.47 | +108.0% |
| **Alexander Volkov** | Jailton Almeida | 68.4% | 2.85 | +92.6% |

*See `UFC321_PREDICTIONS_FULL.txt` for complete breakdown*

## 🔧 Training

Retrain production model with latest data:

```python
python scripts/train_production.py
```

This trains on all data through 2024, validates on 2025 holdout.

## 📈 Backtesting

### Year-by-Year Holdout Validation

```python
python scripts/backtest_yearly_holdouts.py
```

Results:
- 2023: 72.2% accuracy, +157.3% ROI
- 2024: 74.9% accuracy, +179.0% ROI  
- 2025: 70.8% accuracy, +146.9% ROI

### ROI Backtesting with Fixed Stakes

```python
python scripts/backtest_actual_roi_fixed.py
```

## 🧠 Methodology

### Data Leakage Prevention

Uses **FightIQ's proven leakage patterns** to remove 3,931 features containing current-fight information:
- Round-by-round statistics (_r1_, _r2_, etc.)
- Fight outcome variables (winner, method, duration)
- Current fight totals (strikes landed, knockdowns, etc.)

**Keeps safe historical features:**
- Career aggregates (total wins, striking accuracy %, etc.)
- Rolling averages (last 3 fights, last 5 fights)
- Physical attributes (height, reach, age)
- Pre-fight odds

### Feature Engineering

- **Fighter-level features**: Career statistics, momentum indicators
- **Differential features**: Head-to-head comparisons
- **Contextual features**: Weight class, fighting style, home advantage
- **Market features**: Betting odds (7 features when available)

### Model Architecture

**Ensemble of gradient boosting models:**

1. **XGBoost** (max_depth=5, lr=0.023)
2. **LightGBM** (num_leaves=31, lr=0.02)
3. **Simple average** for final predictions

### Conservative Betting Strategy

- **Confidence threshold**: 60%
- **Positive edge required**: Model probability > Market probability
- **Fixed stakes**: $100-200 per bet (1-2 units)
- **No Kelly Criterion**: Avoids compounding for realistic ROI

## 🔑 Configuration

Edit `config/config.yaml` to customize:
- Data paths
- Model hyperparameters
- Betting thresholds
- Feature engineering rules

## 🔌 Odds API Integration

Requires [The Odds API](https://the-odds-api.com/) key:

```python
ODDS_API_KEY = "your_key_here"
```

Set in `scripts/predict_upcoming_ufc321.py` or via environment variable.

## ⚠️ Disclaimer

This system is for **educational and research purposes**. 

- Past performance does not guarantee future results
- Gambling involves risk - bet responsibly
- Always verify predictions against multiple sources
- Consider this one tool among many for informed betting decisions

## 📝 Citation

If you use this system in research or projects, please cite:

```
UFC Master Pipeline (2025)
Production UFC Fight Prediction System
Built with Claude Code by Anthropic
https://github.com/gogs1998/fightiq321claude
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional data sources (FightMetric, Sherdog)
- Advanced feature engineering
- Neural network models
- Live betting integration

## 📄 License

MIT License - see LICENSE file

---

**Built with Claude Code** 🤖 | **Powered by 30+ years of UFC data** 📊 | **Production-ready predictions** 🎯
