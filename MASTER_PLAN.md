# UFC MASTER PIPELINE - MASTER PLAN
### The World's Leading UFC Fight Prediction System

**Version:** 1.0.0
**Created:** October 21, 2025
**Mission:** Build the most accurate, robust, and production-ready UFC prediction pipeline in the world

---

## EXECUTIVE SUMMARY

This master plan consolidates four predecessor repositories into a single, world-class UFC prediction system:

1. **ufc-fight-forecast** → Data infrastructure & scraping
2. **FightIQ** → ML methodology & leakage prevention
3. **FightIQ_improved** → Production-ready models & testing
4. **fightiq_codex** → Unified architecture & automation

**Target Performance:**
- **70%+ accuracy** on unseen holdout data (beats betting markets at 65-68%)
- **15%+ annual ROI** with disciplined Kelly Criterion betting
- **<0.05 calibration error** for trustworthy probabilities
- **Zero data leakage** through automated validation
- **Weekly automated predictions** with Agent Kit orchestration

---

## SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ UFC Stats    │  │ Odds APIs    │  │ Rankings     │              │
│  │ Scrapers     │  │ (The Odds)   │  │ Scrapers     │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                 │                        │
│         └─────────────────┴─────────────────┘                        │
│                           │                                          │
└───────────────────────────┼──────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RAW DATA LAYER (Bronze)                          │
│  BigQuery Tables / Parquet Files                                    │
│  • events_raw  • fights_raw  • fight_stats_raw                      │
│  • odds_raw    • rankings_raw                                       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SILVER DATA LAYER (Cleaned)                       │
│  • Deduplication  • Type normalization  • Missing value handling    │
│  • Timezone fixes • Canonical IDs       • Data validation           │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  GOLD DATA LAYER (Feature Engineered)                │
│  Point-in-Time Safe Features:                                       │
│  • Rolling aggregates (3/5/10/15/20 fights) - STRICT HISTORICAL    │
│  • Matchup differentials (reach, height, age, stance)               │
│  • Betting odds features (vig-free implied prob, logit transforms)  │
│  • Ranking deltas & trends                                          │
│  • Recency metrics (days since fight, fight frequency)              │
│  • Advanced combat metrics (striking efficiency, grappling control) │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      VALIDATION LAYER                                │
│  • Great Expectations suites (data contracts)                       │
│  • Automated leakage detection (3,897+ feature filters)             │
│  • Temporal integrity checks                                        │
│  • Point-in-time join validation                                    │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MODELING LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  BASE MODELS (Optimized with Optuna)                        │   │
│  │  • XGBoost (tuned: max_depth=5, lr=0.023, subsample=0.89)  │   │
│  │  • LightGBM (tuned: num_leaves=31, lr=0.02)                │   │
│  │  • CatBoost (tuned for categorical features)                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                            │                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  ENSEMBLE LAYER (Out-of-Fold Stacking)                      │   │
│  │  • TimeSeriesSplit 5-fold CV                                │   │
│  │  • OOF predictions → Logistic Regression meta-learner       │   │
│  │  • Per-weight-class calibration                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                            │                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  CALIBRATION LAYER                                           │   │
│  │  • Platt scaling (Logistic Regression)                      │   │
│  │  • Isotonic regression (non-parametric)                     │   │
│  │  • Temperature scaling (multi-task)                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                            │                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  MULTI-TASK PREDICTIONS                                      │   │
│  │  • Winner (binary classification)                           │   │
│  │  • Method (KO/Sub/Decision/Other)                           │   │
│  │  • Round (1-5 or Decision)                                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BETTING STRATEGY LAYER                            │
│  • Kelly Criterion position sizing (fractional Kelly: 0.5x)        │
│  • Minimum edge threshold (2%+ expected value)                      │
│  • Probability threshold (only bet if P > 0.55)                     │
│  • Event-level constraints (max 3 bets/event, max 20% exposure)    │
│  • Bankroll management (compound weekly, track drawdown)            │
│  • Policy tuning (grid search on validation set)                    │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BACKTESTING & EVALUATION                          │
│  • Walk-forward validation (3-month windows, 1-month steps)        │
│  • Expanding window (simulates production deployment)               │
│  • Per-fold calibration (inner train/val split)                    │
│  • ROI tracking (annual equity curves, Sharpe ratio, max drawdown) │
│  • Hit rate analysis (by weight class, fighter style, odds range)  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION INFERENCE                              │
│  • Weekly automated predictions (Sundays 2 AM)                      │
│  • Feature alignment & validation                                   │
│  • Multi-model ensemble predictions                                 │
│  • Calibrated probabilities + Kelly bet sizes                       │
│  • CSV/JSON outputs with bet recommendations                        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   MONITORING & ALERTING                              │
│  • Model drift detection (PSI, KL divergence)                       │
│  • Calibration drift monitoring (ECE trends)                        │
│  • Data quality alerts (Great Expectations failures)                │
│  • Performance tracking (rolling accuracy, ROI)                     │
│  • Slack/Email notifications                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## CORE PRINCIPLES

### 1. Zero Data Leakage (Non-Negotiable)
- **Automated detection** of 3,897+ current-fight statistics
- **Point-in-time joins** for all historical features
- **Temporal validation** (no shuffling, strict date ordering)
- **Comprehensive testing** (5+ leakage tests in CI/CD)
- **Great Expectations suites** for data contract enforcement

### 2. Rigorous Validation
- **Temporal splits:** Train (pre-2023), Val (2023-2024), Test (2025+)
- **ONE-TIME holdout test** (never touched until final evaluation)
- **Walk-forward backtesting** (realistic production simulation)
- **Out-of-fold stacking** (no in-sample predictions for meta-learner)

### 3. Production-First Engineering
- **Configuration-driven** (no hardcoded paths/parameters)
- **Modular design** (clear separation of concerns)
- **Comprehensive logging** (loguru with structured outputs)
- **Type hints & docstrings** (Python 3.10+ standards)
- **Automated testing** (pytest with >90% coverage)

### 4. Betting Discipline
- **Kelly Criterion** with fractional sizing (0.5x for safety)
- **Minimum edge** (only bet with 2%+ expected value)
- **Event constraints** (max 3 bets/card, max 20% exposure)
- **Bankroll tracking** (compound growth, drawdown limits)

### 5. Continuous Improvement
- **MLflow experiment tracking** (every training run logged)
- **Weekly retraining** (automated with latest data)
- **Hyperparameter optimization** (Optuna with 500+ trials)
- **A/B testing** (champion/challenger model deployment)

---

## 12-WEEK IMPLEMENTATION ROADMAP

### PHASE 1: FOUNDATION (Weeks 1-2)

**Week 1: Data Infrastructure**
- [ ] Port 10 scrapers from ufc-fight-forecast
  - Events, fighter URLs, fighter data, fight URLs, fight stats
  - Rankings, betting odds (The Odds API)
  - Async/parallel processing, retry logic, rate limiting
- [ ] Set up BigQuery backend (optional, for scale)
- [ ] Implement Parquet-based local storage
- [ ] Create raw data ingestion pipeline
- [ ] Add deduplication & incremental loading

**Week 2: Data Transformation**
- [ ] Build Silver layer transformations
  - Clean fight metadata, normalize odds, parse rankings
  - Type fixes, timezone handling, canonical IDs
- [ ] Build Gold layer feature engineering
  - Rolling aggregates (3/5/10/15/20 fights)
  - Matchup differentials, recency metrics
  - Vig-free odds, ranking deltas
- [ ] Implement point-in-time join validation
- [ ] Create Great Expectations suites (raw/silver/gold)

**Deliverables:**
- ✅ Complete data pipeline (scrape → raw → silver → gold)
- ✅ 8,000+ historical fights with 1,500+ features
- ✅ Automated data quality validation

---

### PHASE 2: MODEL EXCELLENCE (Weeks 3-5)

**Week 3: Baseline Models**
- [ ] Port FightIQ_improved's temporal splitter
- [ ] Implement feature-type-specific imputation
- [ ] Train baseline models (LogReg, XGBoost, LightGBM)
- [ ] Verify parity with FightIQ (69% target)
- [ ] Set up MLflow tracking

**Week 4: Hyperparameter Optimization**
- [ ] Optuna integration (500 trials per model)
- [ ] Grid search for:
  - XGBoost (max_depth, learning_rate, subsample, reg_alpha/lambda)
  - LightGBM (num_leaves, learning_rate, min_data_in_leaf)
  - CatBoost (depth, learning_rate, l2_leaf_reg)
- [ ] Target: 70%+ validation accuracy
- [ ] Save best configurations

**Week 5: Advanced Ensembling**
- [ ] Port FightIQ_improved's OOF stacking
- [ ] TimeSeriesSplit 5-fold CV for OOF predictions
- [ ] Logistic Regression meta-learner
- [ ] Per-weight-class calibration (Platt scaling)
- [ ] Target: 71-72% ensemble accuracy

**Deliverables:**
- ✅ 70%+ validation accuracy (beats betting markets)
- ✅ Properly calibrated probabilities (ECE <0.05)
- ✅ Comprehensive experiment tracking (50+ runs)

---

### PHASE 3: BETTING OPTIMIZATION (Weeks 6-7)

**Week 6: Kelly Criterion Implementation**
- [ ] Port betting strategy from fightiq_codex
- [ ] Implement fractional Kelly (0.5x multiplier)
- [ ] Add minimum edge threshold (2%+)
- [ ] Event-level constraints (max bets, exposure limits)
- [ ] Bankroll tracking with compounding

**Week 7: Policy Tuning & Backtesting**
- [ ] Grid search on validation set:
  - Edge thresholds (1%, 2%, 3%, 5%)
  - Kelly fractions (0.25x, 0.5x, 0.75x, 1x)
  - Probability thresholds (0.50, 0.55, 0.60, 0.65)
- [ ] Walk-forward backtest (2022-2025)
- [ ] Generate annual equity curves
- [ ] Target: 15%+ annual ROI, Sharpe >0.5

**Deliverables:**
- ✅ 15%+ backtested ROI
- ✅ Disciplined betting strategy (max drawdown <25%)
- ✅ Comprehensive bet tracking (CSV outputs)

---

### PHASE 4: MULTI-TASK LEARNING (Weeks 8-9)

**Week 8: Method & Round Predictions**
- [ ] Port multi-task training from fightiq_codex
- [ ] Separate models for:
  - Winner (binary)
  - Method (KO/TKO, Submission, Decision, Other)
  - Round (1-5 or Decision)
- [ ] Temperature scaling for multiclass calibration
- [ ] Target: Method 55%+, Round 60%+

**Week 9: Multi-Task Betting Strategy**
- [ ] Prop bet optimization (KO props, submission props)
- [ ] Parlay builder (winner + method combinations)
- [ ] Expected value calculations for prop bets
- [ ] Policy tuning for method/round bets

**Deliverables:**
- ✅ Multi-task predictions with calibrated probabilities
- ✅ Prop bet recommendations with EV calculations
- ✅ Expanded betting strategy (10+ bet types)

---

### PHASE 5: AUTOMATION & PRODUCTION (Weeks 10-11)

**Week 10: Weekly Orchestration**
- [ ] Agent Kit integration
- [ ] Automated weekly pipeline:
  1. Scrape latest events, fights, odds, rankings (Sunday 2 AM)
  2. Update silver/gold layers
  3. Run Great Expectations validation
  4. Retrain models if needed (monthly)
  5. Generate predictions for upcoming fights
  6. Send Slack/Email notifications
- [ ] Error handling & retry logic
- [ ] Monitoring dashboard (Streamlit optional)

**Week 11: Model Monitoring & Drift Detection**
- [ ] Population Stability Index (PSI) tracking
- [ ] Calibration drift monitoring (ECE trends)
- [ ] Champion/Challenger A/B testing
- [ ] Automated model rollback on performance degradation
- [ ] Alert system (Slack/Email)

**Deliverables:**
- ✅ Fully automated weekly predictions
- ✅ Zero manual intervention required
- ✅ Robust monitoring & alerting

---

### PHASE 6: WORLD-CLASS FEATURES (Week 12)

**Week 12: Advanced Features**
- [ ] Fighter style embeddings (striker/grappler/wrestler)
- [ ] Head-to-head history (date-filtered)
- [ ] Camp/coach features (fighter team analysis)
- [ ] Injury history tracking
- [ ] Weight cut indicators (weigh-in footage analysis optional)
- [ ] Social media sentiment (Twitter/Reddit analysis)
- [ ] Target: Push to 72%+ accuracy

**Bonus: Research Features**
- [ ] Deep learning with fighter embeddings (PyTorch)
- [ ] Attention mechanisms for fight sequences
- [ ] Reinforcement learning for dynamic Kelly sizing
- [ ] Bayesian optimization for hyperparameters

**Deliverables:**
- ✅ 72%+ accuracy on 2025 holdout
- ✅ 20%+ annual ROI
- ✅ Publication-ready methodology

---

## TECHNOLOGY STACK

### Core Data Science
```python
numpy>=1.26.0
pandas>=2.1.0
scipy>=1.11.0
scikit-learn>=1.3.0
```

### Machine Learning
```python
xgboost>=2.0.0
lightgbm>=4.1.0
catboost>=1.2.0
optuna>=3.4.0
```

### Deep Learning (Phase 6)
```python
torch>=2.1.0
pytorch-lightning>=2.1.0
transformers>=4.35.0  # For sentiment analysis
```

### Data Engineering
```python
google-cloud-bigquery>=3.13.0  # Optional
pyarrow>=14.0.0  # Parquet support
great-expectations>=0.18.0
```

### Experiment Tracking
```python
mlflow>=2.9.0
wandb>=0.16.0
```

### Web Scraping
```python
requests>=2.31.0
beautifulsoup4>=4.12.0
aiohttp>=3.9.0  # Async scraping
```

### Utilities
```python
loguru>=0.7.0
pyyaml>=6.0.0
joblib>=1.3.0
pytest>=7.4.0
black>=23.0.0  # Code formatting
ruff>=0.1.0    # Fast linting
```

### Deployment
```python
streamlit>=1.28.0  # Dashboard (optional)
fastapi>=0.104.0   # API (optional)
```

---

## DATA ARCHITECTURE

### Raw Layer (Bronze)
**Storage:** BigQuery tables or Parquet files in `data/raw/`

**Tables:**
1. `events_raw` (columns: event_url, event_name, event_date, event_location)
2. `fights_raw` (columns: fight_url, event_url, fighter_1_url, fighter_2_url, weight_class, title_fight)
3. `fight_stats_raw` (columns: fight_url, fighter_url, knockdowns, strikes_landed, takedowns, etc.)
4. `fighters_raw` (columns: fighter_url, name, height, reach, dob, stance, record)
5. `odds_raw` (columns: fight_url, bookmaker, fighter_1_odds, fighter_2_odds, timestamp)
6. `rankings_raw` (columns: event_date, weight_class, rank, fighter_url)

**Update Frequency:** Daily (incremental)

### Silver Layer (Cleaned)
**Storage:** Parquet files in `data/silver/`

**Transformations:**
- Deduplication (remove duplicate fight_url + fighter_url pairs)
- Type normalization (dates → datetime64, odds → float64)
- Canonical IDs (fighter_url as primary key)
- Missing value policies (median for physical stats, 0 for rolling stats)
- Timezone fixes (all dates → UTC)

**Tables:**
1. `fights_silver.parquet` (8,217 fights, 1994-2025)
2. `odds_silver.parquet` (97,000+ odds snapshots)
3. `rankings_silver.parquet` (91,000+ ranking records)

**Update Frequency:** Daily (after raw ingestion)

### Gold Layer (Feature Engineered)
**Storage:** Parquet file `data/gold/gold_features.parquet`

**Features (1,500+ total):**

**Fighter Physical Attributes:**
- Height (cm), reach (cm), weight (lbs), age (years)
- Stance (Orthodox/Southpaw/Switch)
- BMI, ape index (reach - height)

**Career Statistics (point-in-time safe):**
- Record: wins, losses, draws, no-contests
- Finish rates: KO%, submission%, decision%
- Career striking: SLpM, Str_Acc, SApM, Str_Def
- Career grappling: TD_Avg, TD_Acc, TD_Def, Sub_Avg

**Rolling Window Aggregates (3/5/10/15/20 fights):**
- Per window: strikes landed/attempted, striking accuracy
- Takedowns landed/attempted, takedown accuracy
- Control time, submission attempts
- Knockdowns, significant strikes
- **CRITICAL:** Current fight explicitly excluded from all rolling calcs

**Matchup Differentials:**
- Height differential (F1_height - F2_height)
- Reach differential, age differential
- Win record differential
- Striking differential (F1_SLpM - F2_SLpM)
- Grappling differential (F1_TD_Avg - F2_TD_Avg)

**Betting Odds Features:**
- Decimal odds (converted from American/fractional)
- Implied probability (vig-removed)
- Logit odds (log(p / (1-p)))
- Odds ratio (F1_odds / F2_odds)
- Odds gap (abs(F1_implied_prob - F2_implied_prob))

**Ranking Features:**
- Current rank (as of event date)
- Ranking trend (rank 3 months ago - current rank)
- Title fight indicator
- Ranking differential (F1_rank - F2_rank)

**Recency Metrics:**
- Days since last fight
- Fights in last 12 months
- Average downtime between fights
- Layoff indicator (>365 days since last fight)

**Update Frequency:** Daily (after silver updates)

---

## LEAKAGE PREVENTION FRAMEWORK

### 1. Automated Feature Filters

**Regex patterns to exclude:**
```python
LEAKING_PATTERNS = [
    r'.*_r\d+_.*',           # Round-by-round stats (_r1_, _r2_, etc.)
    r'.*_round_\d+.*',       # Alternative round notation
    r'.*finish_round.*',     # Finish round (only known after fight)
    r'.*finish_time.*',      # Finish time
    r'.*total_strikes.*',    # Total strikes (sum of all rounds)
    r'.*total_.*_att.*',     # Total attempts
    r'.*total_.*_succ.*',    # Total successes
    r'.*fight_duration.*',   # Fight duration (minutes)
    r'.*num_rounds_fought.*',# Rounds fought
    r'.*result.*',           # Result columns
    r'.*winner.*',           # Winner (target variable)
    r'.*method.*',           # Finish method
]
```

**Enforcement:** Automatically applied in `src/data/loaders.py` before any model training.

### 2. Point-in-Time Join Validation

**Rule:** All rolling statistics must use fights with `event_date < current_fight_event_date`.

**Implementation:**
```python
# In feature engineering
fighter_history = fights_df[
    (fights_df['fighter_url'] == current_fighter) &
    (fights_df['event_date'] < current_event_date)
].sort_values('event_date')

rolling_stats = fighter_history.tail(N).agg({...})
```

**Validation:** Unit tests with synthetic data to verify exclusion.

### 3. Great Expectations Suites

**Raw Layer Expectations:**
- `event_date` is not null and is in the past
- `fighter_url` matches URL pattern
- `odds` are positive floats
- No duplicate `(fight_url, fighter_url)` pairs

**Silver Layer Expectations:**
- All raw layer expectations + type constraints
- `height` between 150-220 cm
- `reach` between 150-220 cm
- `age` between 18-50 years
- Odds imply probabilities sum to ~1.0 (after vig removal)

**Gold Layer Expectations:**
- No rolling stat includes current fight
- No future dates in feature calculations
- All features are numeric (no strings)
- No infinite or NaN values in model-ready features

### 4. Temporal Split Enforcement

**Configuration:**
```yaml
splits:
  train_end_date: '2022-12-31'
  val_start_date: '2023-01-01'
  val_end_date: '2024-12-31'
  test_start_date: '2025-01-01'

  shuffle: false  # NEVER shuffle time-series data
  stratify: false # Temporal integrity > class balance
```

**Validation:** Automated check that no test dates appear in training set.

### 5. Comprehensive Testing

**5 Leakage Tests (from FightIQ_improved):**

1. **Current-Fight Statistics Exclusion**
   - Verifies no round-by-round stats in feature set
   - Asserts 3,897+ features correctly excluded

2. **Rolling Statistics Validation**
   - Synthetic fight history with known outcomes
   - Verifies rolling stats exclude current fight

3. **Temporal Ordering**
   - Ensures train dates < val dates < test dates
   - No overlap between splits

4. **Odds Timing Validation**
   - Verifies odds timestamps are pre-fight
   - Detects post-fight odds leakage

5. **Target Leakage**
   - Ensures winner/method/round not in feature set
   - Correlation analysis (no features with >0.95 correlation to target)

**CI/CD:** All tests run on every commit, deployment blocked on failures.

---

## MODEL TRAINING PIPELINE

### 1. Data Loading & Validation
```python
# Load gold features
features_df = load_gold_features()

# Run Great Expectations
validate_data_quality(features_df)

# Automated leakage detection
detect_and_remove_leaky_features(features_df)

# Temporal split
train, val, test = temporal_split(features_df, config)
```

### 2. Feature Engineering & Imputation
```python
# Feature-type-specific imputation
imputer = FeatureTypeImputationStrategy()
train_imputed = imputer.fit_transform(train)
val_imputed = imputer.transform(val)
test_imputed = imputer.transform(test)

# Optional: Remove high-correlation features (>0.90)
selector = CorrelationSelector(threshold=0.90)
train_selected = selector.fit_transform(train_imputed)
val_selected = selector.transform(val_imputed)
test_selected = selector.transform(test_imputed)
```

### 3. Hyperparameter Optimization
```python
# Optuna study (500 trials)
study = optuna.create_study(direction='minimize', sampler=TPESampler())
study.optimize(objective, n_trials=500)

# Best hyperparameters
best_params = study.best_params

# Train with best params
model = XGBClassifier(**best_params)
model.fit(X_train, y_train)
```

### 4. Out-of-Fold Ensemble Stacking
```python
# TimeSeriesSplit (5 folds)
tscv = TimeSeriesSplit(n_splits=5)

# Generate OOF predictions
oof_preds = np.zeros(len(X_train))
for train_idx, val_idx in tscv.split(X_train):
    # Train base models
    xgb_model.fit(X_train[train_idx], y_train[train_idx])
    lgb_model.fit(X_train[train_idx], y_train[train_idx])

    # Predict on validation fold
    oof_preds[val_idx] = (
        xgb_model.predict_proba(X_train[val_idx])[:, 1] * 0.5 +
        lgb_model.predict_proba(X_train[val_idx])[:, 1] * 0.5
    )

# Train meta-learner on OOF predictions
meta_model = LogisticRegression()
meta_model.fit(oof_preds.reshape(-1, 1), y_train)
```

### 5. Calibration
```python
# Platt scaling on validation set
calibrator = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
calibrator.fit(X_val, y_val)

# Evaluate calibration
ece = expected_calibration_error(y_val, calibrator.predict_proba(X_val))
print(f"ECE: {ece:.4f}")  # Target: <0.05
```

### 6. Final Evaluation (ONE-TIME on test set)
```python
# ONLY run after all development is complete
test_preds = calibrator.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, test_preds > 0.5)
logloss = log_loss(y_test, test_preds)
roc_auc = roc_auc_score(y_test, test_preds)

print(f"Test Accuracy: {accuracy:.1%}")  # Target: 70%+
print(f"Test Log Loss: {logloss:.4f}")
print(f"Test ROC AUC: {roc_auc:.4f}")
```

---

## BETTING STRATEGY

### Kelly Criterion Formula
```python
def kelly_fraction(prob, odds, kelly_multiplier=0.5):
    """
    Calculate optimal bet size using Kelly Criterion.

    Args:
        prob: Model probability (0-1)
        odds: Decimal odds (e.g., 2.5)
        kelly_multiplier: Fraction of Kelly to bet (0.5 = half Kelly)

    Returns:
        Fraction of bankroll to bet (0-1)
    """
    edge = prob * odds - 1  # Expected profit per unit bet
    if edge <= 0:
        return 0.0

    kelly = edge / (odds - 1)
    return kelly * kelly_multiplier
```

### Betting Policy

**Constraints:**
- **Minimum edge:** 2% (only bet if expected value > 2%)
- **Probability threshold:** 0.55 (only bet if P(win) > 55%)
- **Kelly fraction cap:** 0.05 (max 5% of bankroll per bet)
- **Kelly multiplier:** 0.5 (half Kelly for safety)
- **Max bets per event:** 3 (diversification)
- **Max exposure per event:** 20% of bankroll (risk management)

**Bankroll Management:**
- Initial bankroll: £1,000
- Compound weekly (reinvest profits)
- Max drawdown stop-loss: -30% (pause betting if reached)

### Walk-Forward Backtesting

**Configuration:**
- Initial training period: 1994-2022 (7,000+ fights)
- Test window: 3 months (rolling)
- Step size: 1 month (overlapping windows)
- Per-fold calibration: 80% train / 20% calibration split

**Evaluation Metrics:**
- Total ROI (%)
- Annual ROI (%)
- Sharpe ratio (returns / volatility)
- Hit rate (% of bets won)
- Max drawdown (%)
- Average bet size (% of bankroll)
- Total units wagered, total profit

**Target Performance:**
- ROI: 15%+ annually
- Sharpe: 0.5+ (0.7+ excellent)
- Hit rate: 60%+ (65%+ excellent)
- Max drawdown: <25%

---

## WEEKLY AUTOMATION PIPELINE

### Sunday 2 AM Schedule

**Step 1: Data Ingestion (2:00-2:30 AM)**
```bash
python scripts/ingest_events.py
python scripts/ingest_fight_urls.py
python scripts/ingest_fight_stats.py
python scripts/ingest_odds.py
python scripts/ingest_rankings.py
```

**Step 2: Data Transformation (2:30-3:00 AM)**
```bash
python scripts/build_silver_fights.py
python scripts/build_silver_odds.py
python scripts/build_silver_rankings.py
python scripts/build_gold_features.py
```

**Step 3: Validation (3:00-3:10 AM)**
```bash
python scripts/run_ge_validations.py
python scripts/validate_data.py
```

**Step 4: Model Retraining (Monthly, 1st Sunday)**
```bash
# Only run on first Sunday of month
if [ $(date +%d) -le 7 ]; then
    python scripts/train_winner_enhanced.py
    python scripts/backtest_walkforward.py
fi
```

**Step 5: Prediction (3:10-3:20 AM)**
```bash
python scripts/predict_upcoming.py --output results/weekly_predictions.csv
```

**Step 6: Notifications (3:20-3:30 AM)**
```bash
python scripts/send_slack_notification.py
python scripts/send_email_report.py
```

### Error Handling
- Retry logic: 3 attempts with exponential backoff
- Fallback: Use previous week's model if retraining fails
- Alerts: Slack/Email on any step failure
- Monitoring: Log all steps to MLflow

---

## MONITORING & DRIFT DETECTION

### Model Performance Tracking

**Metrics Monitored:**
- Rolling accuracy (last 50 predictions)
- Rolling calibration error (ECE on last 100 predictions)
- Rolling ROI (last 3 months)
- Sharpe ratio (last 6 months)

**Alerts:**
- Accuracy drops below 65% for 3 consecutive weeks → Retrain immediately
- ECE exceeds 0.10 → Recalibrate
- ROI turns negative for 2 months → Pause betting, investigate

### Data Drift Detection

**Population Stability Index (PSI):**
```python
def calculate_psi(expected, actual, bins=10):
    """
    Calculate PSI between expected (training) and actual (production) distributions.

    Returns:
        PSI value (0-1+)
        - <0.1: No significant drift
        - 0.1-0.25: Moderate drift
        - >0.25: Severe drift (retrain recommended)
    """
    expected_hist, _ = np.histogram(expected, bins=bins)
    actual_hist, _ = np.histogram(actual, bins=bins)

    expected_pct = expected_hist / len(expected)
    actual_pct = actual_hist / len(actual)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi
```

**Monitoring:**
- Calculate PSI for top 20 features weekly
- Alert if PSI > 0.25 for any feature
- Trigger retraining if 5+ features show drift

### Champion/Challenger A/B Testing

**Deployment Strategy:**
- Champion: Current production model
- Challenger: Newly trained model
- A/B split: 80% Champion, 20% Challenger (for 4 weeks)
- Promotion: If Challenger outperforms (accuracy, ROI), promote to Champion

---

## FILE STRUCTURE

```
UFC-Master-Pipeline/
├── README.md                          # Project overview
├── MASTER_PLAN.md                     # This document
├── QUICKSTART.md                      # 30-minute getting started guide
├── CHANGELOG.md                       # Version history
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation
├── pytest.ini                         # Test configuration
├── .gitignore                         # Git exclusions
│
├── config/
│   ├── config.yaml                    # Main configuration
│   ├── paths.yaml                     # Data/model paths
│   ├── model_params.yaml              # Hyperparameters
│   └── betting_policy.yaml            # Kelly parameters
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py                 # Data loading with leak filters
│   │   ├── splitters.py               # Temporal/walk-forward splits
│   │   ├── preprocessing.py           # Imputation strategies
│   │   └── validation.py              # Data quality checks
│   ├── features/
│   │   ├── __init__.py
│   │   ├── rolling_aggregates.py      # Rolling window stats
│   │   ├── matchup_features.py        # Fighter differentials
│   │   ├── odds_features.py           # Betting odds transformations
│   │   └── ranking_features.py        # Ranking-based features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py                # LogReg, XGB, LGB
│   │   ├── ensemble.py                # OOF stacking
│   │   ├── calibration.py             # Platt/Isotonic calibration
│   │   └── multitask.py               # Winner/Method/Round models
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Classification & betting metrics
│   │   ├── calibration_metrics.py     # ECE, reliability diagrams
│   │   └── backtesting.py             # Walk-forward evaluation
│   ├── betting/
│   │   ├── __init__.py
│   │   ├── kelly_criterion.py         # Position sizing
│   │   └── policy.py                  # Betting constraints
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── scrapers/
│   │   │   ├── __init__.py
│   │   │   ├── ufcstats_events.py     # Event scraper
│   │   │   ├── ufcstats_fighters.py   # Fighter scraper
│   │   │   ├── ufcstats_fights.py     # Fight stats scraper
│   │   │   ├── odds_api.py            # Odds API client
│   │   │   └── rankings.py            # Rankings scraper
│   │   ├── sinks.py                   # BigQuery/Parquet writers
│   │   └── utils.py                   # Deduplication, helpers
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── ge_suites.py               # Great Expectations suites
│   │   └── leakage_detection.py       # Automated leakage tests
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── orchestrator.py            # Weekly automation
│   │   └── monitoring.py              # Drift detection, alerts
│   └── utils/
│       ├── __init__.py
│       ├── config.py                  # Config loader
│       ├── logging.py                 # Loguru setup
│       └── helpers.py                 # Common utilities
│
├── scripts/
│   ├── ingest_events.py               # Scrape UFC events
│   ├── ingest_fight_urls.py           # Scrape fight URLs
│   ├── ingest_fight_stats.py          # Scrape fight statistics
│   ├── ingest_odds.py                 # Fetch betting odds
│   ├── ingest_rankings.py             # Scrape rankings
│   ├── build_silver_fights.py         # Raw → Silver (fights)
│   ├── build_silver_odds.py           # Raw → Silver (odds)
│   ├── build_silver_rankings.py       # Raw → Silver (rankings)
│   ├── build_gold_features.py         # Silver → Gold (features)
│   ├── validate_data.py               # Quick data checks
│   ├── run_ge_validations.py          # Great Expectations runner
│   ├── train_baseline.py              # Baseline model training
│   ├── train_optimized.py             # Hyperparameter tuning
│   ├── train_ensemble.py              # OOF stacking
│   ├── train_multitask.py             # Multi-task learning
│   ├── backtest_walkforward.py        # Walk-forward backtest
│   ├── predict_upcoming.py            # Weekly predictions
│   ├── weekly_orchestrate.py          # Full automation
│   ├── send_notifications.py          # Slack/Email alerts
│   └── analyze_results.py             # Performance analysis
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_loaders.py
│   │   ├── test_features.py
│   │   ├── test_models.py
│   │   └── test_betting.py
│   └── integration/
│       ├── test_leakage.py            # 5 comprehensive leakage tests
│       ├── test_pipeline.py           # End-to-end pipeline test
│       └── test_backtest.py           # Backtesting correctness
│
├── data/
│   ├── raw/                           # Raw scraped data (Bronze)
│   │   ├── events_raw.parquet
│   │   ├── fights_raw.parquet
│   │   ├── fight_stats_raw.parquet
│   │   ├── odds_raw.parquet
│   │   └── rankings_raw.parquet
│   ├── silver/                        # Cleaned data (Silver)
│   │   ├── fights_silver.parquet
│   │   ├── odds_silver.parquet
│   │   └── rankings_silver.parquet
│   └── gold/                          # Feature-engineered (Gold)
│       └── gold_features.parquet
│
├── artifacts/                         # Saved models & artifacts
│   └── <timestamp>/
│       ├── model.pkl
│       ├── imputer.pkl
│       ├── scaler.pkl
│       ├── calibrator.pkl
│       ├── feature_names.json
│       └── config.yaml
│
├── models/                            # Model registry
│   ├── champion/                      # Production model
│   └── challengers/                   # Candidate models
│
├── logs/                              # Training logs
│   └── <timestamp>.log
│
├── results/                           # Predictions & analysis
│   ├── weekly_predictions.csv
│   ├── backtest_roi_report.json
│   ├── equity_curves.png
│   └── feature_importance.csv
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_betting_analysis.ipynb
│
└── docs/                              # Documentation
    ├── API.md                         # API reference
    ├── DATA_DICTIONARY.md             # Feature definitions
    ├── METHODOLOGY.md                 # ML methodology
    └── DEPLOYMENT.md                  # Deployment guide
```

---

## ACCEPTANCE CRITERIA

### Phase 1: Foundation ✅
- [ ] 8,000+ historical fights ingested
- [ ] 10 scrapers ported and tested
- [ ] Silver/Gold layers validated
- [ ] Great Expectations suites passing
- [ ] Zero data quality failures

### Phase 2: Model Excellence ✅
- [ ] 70%+ validation accuracy
- [ ] <0.05 calibration error (ECE)
- [ ] Beats betting market baseline (65-68%)
- [ ] 50+ MLflow experiments tracked
- [ ] Comprehensive model evaluation report

### Phase 3: Betting Optimization ✅
- [ ] 15%+ backtested annual ROI
- [ ] Sharpe ratio >0.5
- [ ] Max drawdown <25%
- [ ] Policy tuning complete (edge/Kelly/probability thresholds)
- [ ] Annual equity curves generated

### Phase 4: Multi-Task Learning ✅
- [ ] Winner accuracy 70%+
- [ ] Method accuracy 55%+
- [ ] Round accuracy 60%+
- [ ] Calibrated probabilities for all tasks
- [ ] Prop bet EV calculations

### Phase 5: Automation ✅
- [ ] Weekly orchestration fully automated
- [ ] Zero manual intervention required
- [ ] Error handling & retry logic tested
- [ ] Monitoring dashboard deployed
- [ ] Slack/Email notifications working

### Phase 6: World-Class ✅
- [ ] 72%+ test accuracy (2025 holdout)
- [ ] 20%+ annual ROI
- [ ] Publication-ready methodology
- [ ] Open-source release (optional)
- [ ] Research paper draft (optional)

---

## RISK MITIGATION

### Technical Risks

**Risk:** Data leakage reintroduced during feature expansion
- **Mitigation:** Automated leakage tests in CI/CD, fail-fast on detection
- **Owner:** Data Engineering Team
- **Severity:** CRITICAL

**Risk:** Model overfitting on validation set
- **Mitigation:** ONE-TIME test set, walk-forward backtesting
- **Owner:** ML Team
- **Severity:** HIGH

**Risk:** Scraper breakage due to website changes
- **Mitigation:** Weekly validation, fallback to cached data, alerts
- **Owner:** Data Engineering Team
- **Severity:** MEDIUM

### Business Risks

**Risk:** Betting market efficiency increases (odds become sharper)
- **Mitigation:** Continuous model improvement, diversify to prop bets
- **Owner:** Strategy Team
- **Severity:** MEDIUM

**Risk:** Insufficient betting volume (limited liquidity)
- **Mitigation:** Multiple sportsbooks, fractional Kelly (smaller bets)
- **Owner:** Operations Team
- **Severity:** LOW

### Operational Risks

**Risk:** Weekly automation fails silently
- **Mitigation:** Comprehensive monitoring, Slack alerts, health checks
- **Owner:** DevOps Team
- **Severity:** MEDIUM

**Risk:** Cloud costs exceed budget
- **Mitigation:** BigQuery query optimization, Parquet compression, budget alerts
- **Owner:** Finance Team
- **Severity:** LOW

---

## SUCCESS METRICS

### Technical KPIs
- **Accuracy:** 70%+ on 2025 holdout (72%+ stretch goal)
- **Calibration:** ECE <0.05 (well-calibrated probabilities)
- **Log Loss:** <0.65 (competitive with Kaggle winners)
- **ROC AUC:** >0.75 (excellent discrimination)

### Betting KPIs
- **Annual ROI:** 15%+ (20%+ stretch goal)
- **Sharpe Ratio:** >0.5 (0.7+ stretch goal)
- **Hit Rate:** 60%+ (65%+ stretch goal)
- **Max Drawdown:** <25% (<20% stretch goal)

### Engineering KPIs
- **Test Coverage:** >90% (pytest)
- **CI/CD Success Rate:** >95% (no broken builds)
- **Deployment Frequency:** Weekly (100% automated)
- **Mean Time to Recovery:** <1 hour (for scraper failures)

### Research KPIs
- **Features Engineered:** 1,500+ (point-in-time safe)
- **Experiments Tracked:** 100+ (MLflow)
- **Models Evaluated:** 10+ (baseline + ensemble + multitask)
- **Hyperparameter Trials:** 1,000+ (Optuna)

---

## TEAM & RESPONSIBILITIES

### Data Engineering
- **Owner:** Scraping, ETL, data quality
- **Deliverables:** Raw/Silver/Gold layers, Great Expectations suites
- **Tools:** Python, BigQuery/Parquet, Great Expectations

### Machine Learning
- **Owner:** Model training, hyperparameter tuning, evaluation
- **Deliverables:** Baseline/ensemble/multitask models, calibration
- **Tools:** XGBoost, LightGBM, Optuna, MLflow

### Betting Strategy
- **Owner:** Kelly Criterion, policy tuning, backtesting
- **Deliverables:** Betting recommendations, ROI reports
- **Tools:** NumPy, pandas, Kelly formula

### DevOps
- **Owner:** Automation, monitoring, deployment
- **Deliverables:** Weekly orchestration, drift detection, alerts
- **Tools:** Agent Kit, Slack API, Streamlit

### Research
- **Owner:** Advanced features, deep learning, experimentation
- **Deliverables:** Fighter embeddings, sentiment analysis, publications
- **Tools:** PyTorch, Transformers, Jupyter

---

## NEXT STEPS

### Immediate (This Week)
1. ✅ Create project directory structure
2. ✅ Write master plan (this document)
3. [ ] Port 10 scrapers from ufc-fight-forecast
4. [ ] Set up BigQuery backend (or Parquet fallback)
5. [ ] Create initial Great Expectations suites

### Short-Term (Weeks 1-4)
1. [ ] Complete data pipeline (raw → silver → gold)
2. [ ] Port ML core from FightIQ_improved
3. [ ] Verify parity (69% accuracy target)
4. [ ] Hyperparameter optimization (70%+ target)
5. [ ] OOF ensemble stacking

### Medium-Term (Weeks 5-8)
1. [ ] Kelly Criterion betting strategy
2. [ ] Walk-forward backtesting (15%+ ROI target)
3. [ ] Multi-task learning (winner/method/round)
4. [ ] Policy tuning on validation set

### Long-Term (Weeks 9-12)
1. [ ] Weekly automation with Agent Kit
2. [ ] Model monitoring & drift detection
3. [ ] Advanced features (embeddings, sentiment)
4. [ ] Final test set evaluation (72%+ target)
5. [ ] Production deployment

---

## APPENDIX

### A. Data Sources
1. **UFCStats.com** - Primary fight statistics
2. **The Odds API** - Betting odds (multiple bookmakers)
3. **UFC.com** - Official rankings
4. **ESPN** - Fighter profiles (optional)
5. **Tapology** - Fighter records (optional)

### B. Recommended Reading
1. "Advances in Financial Machine Learning" - Marcos López de Prado
2. "Forecasting: Principles and Practice" - Hyndman & Athanasopoulos
3. "The Kelly Criterion in Blackjack Sports Betting" - Edward O. Thorp
4. "Machine Learning for Asset Managers" - Marcos López de Prado
5. UFC prediction Kaggle competitions (search "UFC" on Kaggle)

### C. Benchmark Comparisons
- **Betting Markets:** 65-68% accuracy (our baseline to beat)
- **Academic Papers:** ~65% accuracy (Raschka et al., 2020)
- **Kaggle Competitions:** 67-72% accuracy (top leaderboard)
- **FightIQ (predecessor):** 69% accuracy on 2025 holdout
- **Our Target:** 72%+ accuracy with 20%+ ROI

### D. Contact & Support
- **Project Lead:** [Your Name]
- **Data Engineering:** [Team Contact]
- **ML Engineering:** [Team Contact]
- **Betting Strategy:** [Team Contact]
- **Repository:** `D:\Codex\UFC-Master-Pipeline\`
- **MLflow UI:** `http://localhost:5000`
- **Streamlit Dashboard:** `http://localhost:8501` (optional)

---

## CHANGELOG

**Version 1.0.0 (October 21, 2025)**
- Initial master plan created
- Unified architecture from 4 predecessor repos
- 12-week implementation roadmap
- Comprehensive technical specifications
- Acceptance criteria defined

---

**THIS IS THE WORLD'S LEADING UFC PREDICTION PIPELINE.**

**Let's build it. 🥊**
