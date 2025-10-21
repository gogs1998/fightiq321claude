# Kaggle Notebook Guide

## UFC Master Pipeline - Production System Notebook

This guide explains how to use the `UFC_Master_Pipeline_Production_System.ipynb` notebook on Kaggle.

---

## Quick Start on Kaggle

### 1. Upload the Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click **"New Notebook"** → **"File"** → **"Upload Notebook"**
3. Upload `UFC_Master_Pipeline_Production_System.ipynb`

### 2. Add Required Dataset

You need the FightIQ UFC dataset to run the full notebook:

**Option A: Use Kaggle Dataset (Recommended)**
1. In your notebook, click **"Add Data"** in the right panel
2. Search for: `"FightIQ UFC Fight Data 1993-2025"`
3. Add the dataset: https://www.kaggle.com/datasets/asaniczka/fightiq-ufc-fight-data-1993-2025

**Option B: Upload Your Own**
1. Download from: https://www.kaggle.com/datasets/asaniczka/fightiq-ufc-fight-data-1993-2025
2. Upload `UFC_full_data_golden.csv` to Kaggle as a dataset

### 3. Run the Notebook

Once the dataset is added, the notebook will automatically detect it at `/kaggle/input/...`

Simply click **"Run All"** or run cells sequentially!

---

## Notebook Structure

### Section 1: Project Overview
- Introduction to the UFC Master Pipeline
- Key achievements and innovations
- Model architecture overview

### Section 2: Key Features & Innovation
- Data leakage prevention (FightIQ patterns)
- Feature engineering breakdown
- Betting strategy explanation

### Section 3: Setup & Requirements
- Library installation and imports
- Environment setup

### Section 4: Data Loading & Preprocessing
- Dataset loading from Kaggle input
- Data leakage detection and removal
- Missing value handling

### Section 5: Exploratory Data Analysis
- Dataset statistics and temporal patterns
- Feature correlation analysis
- Visualization of key trends

### Section 6: Model Training (Production Pipeline)
- Temporal data split (1994-2024 → 2025)
- XGBoost training
- LightGBM training
- Ensemble creation

### Section 7: Model Evaluation & Performance
- Accuracy, AUC, Log Loss metrics
- Confusion matrices
- Feature importance analysis
- Classification reports

### Section 8: ROI Backtesting with Real Odds
- Conservative, Moderate, Aggressive strategies
- Cumulative profit curves
- Win/Loss analysis
- Expected ROI calculations

### Section 9: Production Predictions (UFC 321)
- Real-world deployment example
- High-confidence betting recommendations
- Underdog value plays
- Prediction visualizations

### Section 10: Results & Conclusions
- Performance summary
- Key achievements
- Limitations and future work
- Acknowledgments

---

## Expected Outputs

When you run the notebook, you'll see:

### Performance Metrics
- **Test Accuracy**: ~70.8% on 2025 holdout
- **Test AUC**: ~0.7292
- **Backtested ROI**: +146.9% (Conservative strategy)
- **Win Rate**: ~75.8% on high-confidence bets

### Visualizations
- 15+ professional plots and charts
- Confusion matrices for all models
- ROI performance curves
- Feature importance rankings
- Prediction confidence distributions

### Predictions
- 26 UFC 321 fight predictions
- 18 high-confidence betting recommendations
- Expected ROI for each bet
- Top underdog value plays

---

## Customization Options

### Change Betting Strategy

In Section 8, modify the confidence threshold:

```python
# Conservative (default)
results = backtest_roi(..., confidence_threshold=0.60)

# More aggressive
results = backtest_roi(..., confidence_threshold=0.55)

# Very aggressive
results = backtest_roi(..., confidence_threshold=0.50)
```

### Use Different Date Splits

In Section 6, adjust the temporal split:

```python
# Current: 1994-2024 training, 2025 testing
test_start_date = '2025-01-01'

# Alternative: 1994-2023 training, 2024-2025 testing
test_start_date = '2024-01-01'
```

### Hyperparameter Tuning

Modify XGBoost parameters in Section 6:

```python
xgb_params = {
    'max_depth': 5,        # Tree depth (3-10)
    'learning_rate': 0.05, # Learning rate (0.01-0.2)
    'n_estimators': 300,   # Number of trees (100-500)
    'subsample': 0.8,      # Sample ratio (0.6-1.0)
    # ... other params
}
```

---

## Troubleshooting

### Issue: Dataset Not Found

**Error**: `FileNotFoundError: UFC_full_data_golden.csv not found`

**Solution**:
1. Make sure you added the FightIQ dataset via Kaggle's "Add Data" feature
2. Check the dataset is at `/kaggle/input/fightiq-ufc-fight-data-1993-2025/`
3. Update the path in Section 4 if needed

### Issue: Out of Memory

**Error**: `MemoryError` or kernel crash

**Solution**:
1. Enable GPU/TPU in Kaggle settings (Settings → Accelerator → GPU)
2. Reduce feature count in Section 4:
   ```python
   # Select top N features only
   top_features = ensemble_importance.head(500)['feature'].tolist()
   X_train = X_train[top_features]
   X_test = X_test[top_features]
   ```

### Issue: Slow Execution

**Solution**:
1. Use GPU accelerator (Settings → GPU)
2. Reduce training data size:
   ```python
   # Use only recent years
   train_data = df[(df['event_date'] >= '2015-01-01') &
                   (df['event_date'] < test_start_date)]
   ```

### Issue: Missing UFC 321 Predictions

**Error**: `predictions_ufc321.csv not found`

**Solution**:
This file is from the production pipeline. For Kaggle, you can:
1. Upload `predictions_ufc321.csv` as a separate dataset
2. Or skip Section 9 (it's for demonstration only)
3. Or generate predictions from scratch using the prediction pipeline

---

## Publishing on Kaggle

### Make Your Notebook Public

1. Click **"Share"** in the top-right
2. Select **"Public"**
3. Add a description and tags:
   - Tags: `ufc`, `sports-analytics`, `machine-learning`, `betting`, `xgboost`, `lightgbm`
   - Title: `UFC Master Pipeline: 70.8% Accuracy + 146% ROI`

### Best Practices

1. **Run All Cells**: Make sure notebook runs without errors
2. **Save Output**: Keep visualizations visible (don't clear output)
3. **Add Comments**: Explain any custom modifications
4. **Credit Sources**: Mention FightIQ dataset and original authors

### Recommended Description

```
UFC Fight Prediction System with Production Results

- 70.8% accuracy on 2025 holdout (unseen data)
- +146.9% ROI on backtested betting strategy
- 1,476 leak-free features from 31 years of UFC data
- Ensemble: XGBoost + LightGBM
- Includes real UFC 321 predictions

Complete production pipeline from data preprocessing to
live predictions. Uses rigorous temporal validation and
real historical betting odds for ROI validation.

Dataset: FightIQ UFC Fight Data (1993-2025)
```

---

## Performance Benchmarks

Typical execution times on Kaggle (with GPU):

| Section | Time |
|---------|------|
| Data Loading | ~30 sec |
| Preprocessing | ~1 min |
| EDA | ~2 min |
| XGBoost Training | ~5 min |
| LightGBM Training | ~4 min |
| Evaluation | ~1 min |
| ROI Backtesting | ~2 min |
| Visualizations | ~1 min |
| **Total** | **~17 min** |

---

## Additional Resources

- **GitHub Repository**: https://github.com/gogs1998/fightiq321claude
- **Dataset Source**: https://www.kaggle.com/datasets/asaniczka/fightiq-ufc-fight-data-1993-2025
- **FightIQ Documentation**: [FightIQ Leakage Patterns](https://github.com/balaustrada/fightIQ)

---

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the GitHub repository README
3. Comment on the Kaggle notebook
4. Open an issue on GitHub: https://github.com/gogs1998/fightiq321claude/issues

---

## License

MIT License - Feel free to fork, modify, and build upon this work!

---

**Happy Modeling! 🥊📊**
