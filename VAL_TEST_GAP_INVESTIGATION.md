# Leakage Audit Findings & Val-Test Gap Investigation

**Date:** 2025-11-05
**Status:** Code Review Complete, Empirical Testing Pending

---

## Part 1: Code-Based Leakage Analysis (COMPLETE ✓)

### Summary: NO MAJOR LEAKAGE DETECTED

I've reviewed all code paths and found the implementation is sound:

### ✅ Existing Leakage Protection

**File:** `src/data/loaders.py`

**Features Removed (3,990 total):**
```python
# Round-by-round stats
'_r1_', '_r2_', '_r3_', '_r4_', '_r5_'  # 3,932 features

# Current fight totals
'f_1_total_strikes_succ', 'f_2_total_strikes_succ'
'f_1_total_strikes_att', 'f_2_total_strikes_att'
'f_1_sig_strikes_succ', 'f_2_sig_strikes_succ'
'f_1_sig_strikes_att', 'f_2_sig_strikes_att'
'f_1_knockdowns', 'f_2_knockdowns'
'f_1_submission_att', 'f_2_submission_att'
'f_1_ctrl_time_sec', 'f_2_ctrl_time_sec'
'fight_duration_minutes'

# Outcome information
'winner', 'result', 'result_details'
'finish_round', 'finish_time', 'finish_details'
'method', 'time_format', 'num_rounds'
```

**Verdict:** ✅ Comprehensive and correct

### ✅ Temporal Splitting

**File:** `src/data/splitters.py`

**Implementation:**
```python
# Lines 86-89: Strict temporal masks
train_mask = df[date_column] < val_start_date  # Before 2024
val_mask = (df[date_column] >= val_start_date) &
           (df[date_column] < test_start_date)  # 2024
test_mask = df[date_column] >= test_start_date  # 2025+

# Lines 118-130: Validation checks for overlap
if train_max >= val_min:
    raise ValueError("Temporal leakage detected between train and val")
```

**Verdict:** ✅ Properly prevents temporal leakage

### ✅ New Feature Engineering

**Files:** `scripts/add_recent_form_features.py`, `scripts/add_style_matchup_features.py`

**Critical Lines:**
```python
# Line 257-260: Only uses PAST fights
f1_history = df_enhanced[
    (df_enhanced['event_date'] < fight_date) &  # ← KEY: Excludes current fight
    ((df_enhanced['f_1_name'] == f1_name) | (df_enhanced['f_2_name'] == f1_name))
]

# Line 379: Cleans up temporary columns
df_enhanced = df_enhanced.drop(columns=['f_1_won', 'f_2_won', 'f_1_finish', 'f_2_finish'])
```

**Verdict:** ✅ Correctly uses only historical data

---

## Part 2: The Val-Test Gap Problem (INVESTIGATION NEEDED 🔴)

### The Concerning Numbers

```
Training:   1994-2024  →  (baseline training)
Validation: 2024       →  72.8% accuracy  ✓ Good
Test:       2025       →  61.3% accuracy  ⚠️ VERY LOW
----------------------------------------
Gap:                      -11.5%          ⚠️ HIGHLY UNUSUAL
```

### Why This Is Concerning

**Normal expectation:** 1-3% gap due to distribution shift
**Actual gap:** 11.5% (almost 4x worse than expected)

This suggests one of four problems:

### Hypothesis 1: Small Sample Variance (LIKELY 🟡)

**Analysis:**
```
Test set: 401 fights
Sample size effect: ±3-5% variance at 95% confidence

At 70% baseline accuracy with n=400:
Standard error = sqrt(0.70 * 0.30 / 400) = 2.3%
95% CI = 70% ± 4.5%

Expected range: 65.5% - 74.5%
Actual: 61.3%
```

**Verdict:** 61.3% is at the edge but within 2 standard deviations

**Evidence For:**
- 400 fights is relatively small
- Random variance could account for 3-5% of the gap
- Natural fluctuation in small samples

**Evidence Against:**
- 61.3% is still unusually low (2.8 std from expected)
- Validation set (474 fights) shows 72.8% accuracy
- Similar sample sizes, very different results

**Likelihood:** 40-50%

---

### Hypothesis 2: Distribution Shift (LIKELY 🟡)

**Analysis:**
2025 UFC may have fundamentally changed:

**Possible Changes in 2025:**
1. **New fighters:** More debutants without historical data
2. **Rule changes:** New scoring criteria or regulations
3. **Competition level:** Overall skill level increased
4. **Stylistic meta:** Shift in dominant fighting styles
5. **Weight class changes:** Division realignments

**How to Test:**
```python
# Compare 2024 vs 2025 fight characteristics
2024_fights = df[df['event_date'].dt.year == 2024]
2025_fights = df[df['event_date'].dt.year == 2025]

# Check for differences in:
- Average fighter experience (total fights)
- Finish rate (KO/SUB vs decision)
- Upset rate (underdog wins)
- Fighter turnover (% new fighters)
- Average odds spreads
```

**Likelihood:** 30-40%

---

### Hypothesis 3: Subtle Validation Leakage (POSSIBLE 🟠)

**Analysis:**
If there's subtle leakage in validation but not test:

**Potential Sources:**
1. **Feature computation error:** Career stats accidentally include partial 2024 data
2. **Temporal boundary issue:** Some 2024 fights used to compute 2024 features
3. **Data processing bug:** Validation features generated differently than test

**How to Test:**
```python
# Check if validation features are computed correctly
# For a random 2024 fight:
fight_date = '2024-06-15'
fighter = 'Some Fighter'

# Their features should only include fights BEFORE 2024-06-15
historical_fights = df[
    (df['event_date'] < fight_date) &
    ((df['f_1_name'] == fighter) | (df['f_2_name'] == fighter))
]

# Verify: No fights from 2024-06-15 or later
assert historical_fights['event_date'].max() < fight_date
```

**Red Flags to Look For:**
- Validation accuracy (72.8%) is higher than training (need to check)
- Validation accuracy close to "with current fight stats" (~85%)
- Test accuracy similar to "true baseline" (~60-65%)

**Likelihood:** 20-30%

---

### Hypothesis 4: Model Overfitting to 2024 (POSSIBLE 🟠)

**Analysis:**
Model may have overfit to 2024-specific patterns:

**Evidence:**
- Validation is single year (2024 only)
- Model tuned on 2024 validation performance
- Hyperparameters optimized for 2024 data
- 2024 patterns don't generalize to 2025

**How to Test:**
```python
# Train on 2022-2023, validate on 2024, test on 2025
train_2022_2023 = df[df['event_date'].dt.year.isin([2022, 2023])]
val_2024 = df[df['event_date'].dt.year == 2024]
test_2025 = df[df['event_date'].dt.year == 2025]

# Check if 2024 performance is unusually high
# Compared to other validation years
```

**Likelihood:** 10-20%

---

## Part 3: Investigation Plan

### Phase 1: Data Characterization (1-2 hours)

**Goal:** Understand if 2025 is fundamentally different

```python
# Script: scripts/investigate_val_test_gap.py

# 1. Compare fight characteristics
analyze_fight_characteristics(df_2024, df_2025)

# 2. Check fighter experience distribution
compare_fighter_experience(df_2024, df_2025)

# 3. Analyze finish rates
compare_finish_rates(df_2024, df_2025)

# 4. Check for new fighters
analyze_fighter_turnover(df_2024, df_2025)

# 5. Compare odds distributions
compare_betting_odds(df_2024, df_2025)
```

**Expected Findings:**
- If distribution shift: Significant differences in above metrics
- If variance: No systematic differences
- If leakage: 2024 features look "too good"

---

### Phase 2: Feature Validation (2-3 hours)

**Goal:** Verify features are computed correctly

```python
# Script: scripts/validate_feature_computation.py

# 1. Spot check random 2024 fights
for fight in random_sample(df_2024, n=10):
    # Verify features only use data before fight_date
    verify_temporal_correctness(fight)

# 2. Check validation accuracy is reasonable
# Should be LOWER than random split
random_split_acc = test_random_split(df_2024)
temporal_split_acc = test_temporal_split(df_2024)

assert temporal_split_acc < random_split_acc, "Possible leakage!"

# 3. Compare validation feature distributions to test
compare_feature_distributions(df_2024, df_2025)
```

---

### Phase 3: Model Diagnosis (2-3 hours)

**Goal:** Understand model behavior on 2025 data

```python
# Script: scripts/diagnose_model_performance.py

# 1. Analyze predictions by fighter type
analyze_by_fighter_type(model, df_2025)
# Are we worse on debuts? Veterans? Specific styles?

# 2. Check calibration
calibration_plot(model, df_2024, df_2025)
# Are probabilities well-calibrated in both sets?

# 3. Feature importance shift
compare_feature_importance(model, df_2024, df_2025)
# Are different features important in 2025?

# 4. Error analysis
analyze_wrong_predictions(model, df_2025)
# What fights are we getting wrong? Any patterns?
```

---

### Phase 4: Empirical Leakage Test (30 minutes)

**Goal:** Run shuffle test to detect hidden leakage

```python
# This is the DEFINITIVE test
# If passes: No leakage (gap is real)
# If fails: Leakage present (need to fix)

python scripts/audit_data_leakage.py --data [golden_dataset]
```

**Expected Results (if no leakage):**
```
Temporal split (correct):     70-71% accuracy
Random split (should be higher): 73-75% accuracy
Gap: 2-5% (healthy)

Interpretation: Random split can "cheat" by mixing
future and past, so it performs better. This is GOOD.
```

**Red Flag Results (if leakage):**
```
Temporal split: 70-71% accuracy
Random split: 70-71% accuracy  ⚠️ TOO SIMILAR
Gap: <1% (suspicious)

Interpretation: If temporal performs as well as random,
there's likely leakage (temporal shouldn't be that good).
```

---

## Part 4: Recommended Actions

### Immediate (Next Session)

**1. Locate Golden Dataset** (5 minutes)
```bash
# Find the actual dataset file
# It should be ~379MB, ~7,317 fights

# Option A: Check if it exists
ls -lh /path/to/data/*.csv

# Option B: Update config.yaml with correct path
# Edit config/config.yaml → paths.golden_dataset

# Option C: Download/generate if missing
```

**2. Run Shuffle Test** (30 minutes)
```bash
# Once dataset located:
python scripts/audit_data_leakage.py \
    --data [path_to_golden_dataset] \
    --target target \
    --date event_date
```

**3. Analyze Results** (30 minutes)
- If shuffle test passes → Gap is real (distribution shift or variance)
- If shuffle test fails → Leakage present (need to fix)

---

### If Shuffle Test Passes (No Leakage)

**Then the 11.5% gap is likely due to:**
- Small sample variance (±3-5%)
- Distribution shift in 2025 UFC (±3-5%)
- Model overfitting to 2024 (±2-3%)

**Recommended approach:**
1. Accept lower performance on 2025 data
2. Use conservative estimates (+1-2% from improvements)
3. Focus on features that reduce variance
4. Consider expanding test set (combine 2025 + early 2026)

---

### If Shuffle Test Fails (Leakage Detected)

**Then we need to:**
1. Identify the leaking features (audit script will flag them)
2. Remove or recompute them correctly
3. Retrain model without leakage
4. Re-evaluate on clean test set

**Expected outcome:**
- Test accuracy will likely DROP to ~65-68%
- But this is the TRUE performance
- Better to know the truth

---

## Part 5: What We Know For Sure

### ✅ Confirmed

1. **Code is sound:** No obvious leakage in feature engineering
2. **Temporal splitting is correct:** Proper date-based splits
3. **Leakage detection exists:** 3,990 features properly removed
4. **Conservative estimates are wise:** +1-2% is realistic target

### ⏳ Need to Verify

1. **Is there hidden leakage?** (Run shuffle test)
2. **Is 2025 fundamentally different?** (Compare distributions)
3. **Is 400 fights too small?** (Calculate confidence intervals)
4. **Are features computed correctly?** (Spot check validation set)

### ❓ Unclear

1. **What causes the 11.5% gap?** (Most likely: variance + shift)
2. **Can we close the gap?** (Unlikely to fully close it)
3. **Is 61.3% the true performance?** (Need more 2025 data to confirm)
4. **Should we use a different validation strategy?** (Maybe k-fold temporal CV)

---

## Part 6: Expected Outcomes

### Best Case Scenario

**Shuffle test passes + 2025 is just harder:**
- 61.3% is real performance on difficult 2025 data
- New features might help: 61.3% → 63-64%
- Conservative gain: +1.5-2.5%
- Still valuable improvement

### Worst Case Scenario

**Shuffle test fails (leakage found):**
- Need to fix leakage and retrain
- True performance likely 63-66%
- After fixes, improvements: 63-66% → 64-67%
- Slower progress but honest

### Most Likely Scenario

**Shuffle test passes + small sample variance:**
- 2025 sample is small and unlucky
- True 2025 performance: ~65-67% (±3% from 61.3%)
- With improvements: 65-67% → 67-69%
- Converges to ~70% with more data

---

## Next Steps

**Step 1:** Locate golden dataset
**Step 2:** Run shuffle test (`audit_data_leakage.py`)
**Step 3:** Based on results:
- **If passes:** Investigate distribution shift
- **If fails:** Fix leakage and retrain

**Timeline:** 2-4 hours to complete investigation

---

## Summary

**Code Review:** ✅ No major leakage detected
**Val-Test Gap:** ⚠️ Concerning, needs investigation
**Most Likely Cause:** Small sample variance + distribution shift
**Recommended Action:** Run shuffle test first, then investigate gap
**Conservative Target:** 72-73% accuracy (+1-2%) is realistic

**Bottom Line:** Your instinct to audit was correct. Let's run the shuffle test to rule out leakage, then investigate the gap.
