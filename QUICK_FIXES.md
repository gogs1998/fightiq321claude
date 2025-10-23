# Quick Fixes - Critical Issues

This document provides immediate fixes for the critical issues identified in the system assessment.

---

## Fix 1: Configuration Paths (Windows → Cross-platform)

### Issue
Hard-coded Windows paths in `config/config.yaml`:
```yaml
data_root: "D:/Codex/UFC-Master-Pipeline/data"
```

### Solution
Update `config/config.yaml` to use relative paths and environment variables:

```yaml
# UFC Master Pipeline - Main Configuration
# All paths, parameters, and settings in one place

# Project Information
project:
  name: "UFC Master Pipeline"
  version: "1.0.0"
  description: "World's leading UFC fight prediction system"

# Paths Configuration
paths:
  # Data directories (relative to project root)
  data_root: "${DATA_ROOT:./data}"
  raw_data: "${DATA_ROOT:./data}/raw"
  silver_data: "${DATA_ROOT:./data}/silver"
  gold_data: "${DATA_ROOT:./data}/gold"

  # Model artifacts
  models_dir: "./models"
  artifacts_dir: "./artifacts"

  # Results & logs
  results_dir: "./results"
  logs_dir: "./logs"

  # MLflow
  mlflow_tracking_uri: "file://./mlruns"

  # Primary datasets
  golden_dataset: "${DATA_ROOT:./data}/UFC_full_data_golden.csv"
  silver_dataset: "${DATA_ROOT:./data}/UFC_full_data_silver.csv"
  odds_dataset: "${DATA_ROOT:./data}/UFC_betting_odds.csv"
  rankings_dataset: "${DATA_ROOT:./data}/UFC_rankings_history.csv"
```

### Environment Variables
Create `.env` file:
```bash
# .env
DATA_ROOT=/path/to/data
ODDS_API_KEY=your_api_key_here
```

---

## Fix 2: Error Handling in Data Loading

### Issue
No error handling in `src/data/loaders.py`:
```python
df = pd.read_csv(data_path)
```

### Solution
Add comprehensive error handling:

```python
def load_ufc_data(
    data_path: str = None,
    remove_leaking_features: bool = True,
    max_rows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load UFC dataset with automatic leak detection.
    
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data file is empty or invalid
        Exception: For other loading errors
    """
    config = get_config()
    
    if data_path is None:
        data_path = config.paths.golden_dataset
    
    # Validate file exists
    data_file = Path(data_path)
    if not data_file.exists():
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Please ensure the data file exists or update config.yaml"
        )
    
    # Validate file size
    if data_file.stat().st_size == 0:
        raise ValueError(f"Data file is empty: {data_path}")
    
    logger.info("="*80)
    logger.info("LOADING UFC DATA")
    logger.info("="*80)
    logger.info(f"Data path: {data_path}")
    
    try:
        # Load data
        df = pd.read_csv(data_path, nrows=max_rows)
        logger.info(f"✓ Loaded {len(df):,} fights, {len(df.columns)} columns")
        
        # Validate required columns
        required_cols = ['event_date']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
    except pd.errors.EmptyDataError:
        raise ValueError(f"Data file is empty or invalid: {data_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse CSV file: {e}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    # Rest of function...
    return df
```

---

## Fix 3: Test Fixtures for Missing Data

### Issue
Tests fail because data files are missing:
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/UFC_full_data_golden.csv'
```

### Solution 1: Skip tests when data is missing

Update `tests/integration/test_leakage.py`:

```python
import pytest
import pandas as pd
from pathlib import Path

DATA_FILE = Path('data/UFC_full_data_golden.csv')

@pytest.fixture
def skip_if_no_data():
    """Skip test if data file is not available"""
    if not DATA_FILE.exists():
        pytest.skip(f"Data file not found: {DATA_FILE}")

def test_no_target_leakage(skip_if_no_data):
    """Test that target variables are not in features"""
    df = pd.read_csv(DATA_FILE)
    # Rest of test...

def test_rolling_stats_manual_audit_real_data(skip_if_no_data):
    """Manually verify rolling stats on REAL data"""
    df = pd.read_csv(DATA_FILE, parse_dates=['event_date'])
    # Rest of test...
```

### Solution 2: Create synthetic test data

Create `tests/fixtures/test_data.py`:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_test_ufc_data(n_fights=100) -> pd.DataFrame:
    """Generate synthetic UFC fight data for testing"""
    
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i*7) for i in range(n_fights)]
    
    # Generate fighters
    fighters = [f"Fighter_{i}" for i in range(50)]
    
    data = []
    for i, date in enumerate(dates):
        f1, f2 = np.random.choice(fighters, 2, replace=False)
        
        fight = {
            'event_date': date,
            'f_1': f1,
            'f_2': f2,
            'winner': np.random.choice([0, 1]),
            
            # Historical features (safe)
            'f_1_wins_total': np.random.randint(5, 20),
            'f_2_wins_total': np.random.randint(5, 20),
            'f_1_height_cm': np.random.randint(170, 200),
            'f_2_height_cm': np.random.randint(170, 200),
            'f_1_reach_cm': np.random.randint(170, 200),
            'f_2_reach_cm': np.random.randint(170, 200),
            
            # Rolling features (safe)
            'f_1_avg_sig_strikes_last_3': np.random.uniform(50, 100),
            'f_2_avg_sig_strikes_last_3': np.random.uniform(50, 100),
            
            # Leaking features (should be removed)
            'f_1_sig_strikes_succ': np.random.randint(50, 150),  # Current fight!
            'f_2_sig_strikes_succ': np.random.randint(50, 150),  # Current fight!
            'body_acc_r1_f_1': np.random.uniform(0, 1),  # Round data!
            'finish_round': np.random.choice([1, 2, 3, 4, 5]),  # Outcome!
            'f_1_odds': np.random.uniform(1.2, 3.0),  # Betting odds
        }
        data.append(fight)
    
    return pd.DataFrame(data)

def save_test_data():
    """Save test data to fixtures directory"""
    Path('tests/fixtures').mkdir(exist_ok=True)
    df = generate_test_ufc_data()
    df.to_csv('tests/fixtures/test_ufc_data.csv', index=False)
    print(f"✓ Created test data: {len(df)} fights")

if __name__ == '__main__':
    save_test_data()
```

Then update tests to use this fixture:

```python
import pytest
from pathlib import Path
from tests.fixtures.test_data import generate_test_ufc_data

@pytest.fixture
def test_data():
    """Provide test data for all tests"""
    return generate_test_ufc_data(n_fights=100)

def test_no_target_leakage(test_data):
    """Test that target variables are not in features"""
    from src.data.loaders import load_ufc_data, _is_current_fight_stat
    
    # Use test data instead of real data
    df = test_data
    
    # Remove leaking features
    safe_features = [col for col in df.columns 
                    if not _is_current_fight_stat(col)]
    
    # Verify target not in features
    assert 'winner' not in safe_features
    assert 'finish_round' not in safe_features
```

---

## Fix 4: Document Data Requirements

Create `DATA_SETUP.md`:

```markdown
# Data Setup Guide

## Required Data Files

The UFC Master Pipeline requires the following data files:

### 1. UFC_full_data_golden.csv

**Location**: `data/UFC_full_data_golden.csv`  
**Size**: ~100 MB  
**Rows**: ~8,000 fights  
**Columns**: ~5,000 features  

**Source**: FightIQ dataset (private)

**Alternative Sources**:
- UFC Stats API: https://www.ufcstats.com/
- Kaggle UFC Dataset: https://www.kaggle.com/datasets/
- Custom scraping (see scripts/)

**Required Columns**:
- `event_date` (datetime) - Fight date
- `f_1` (string) - Fighter 1 name
- `f_2` (string) - Fighter 2 name
- `winner` (int) - Winner (0 or 1)
- Fighter statistics (see schema below)

### 2. UFC_betting_odds.csv

**Location**: `data/UFC_betting_odds.csv`  
**Size**: ~10 MB  
**Source**: Best Fight Odds (https://www.bestfightodds.com/)

### 3. UFC_rankings_history.csv

**Location**: `data/UFC_rankings_history.csv`  
**Size**: ~5 MB  
**Source**: UFC Rankings (https://www.ufc.com/rankings)

## Data Schema

See `docs/data_schema.md` for complete schema documentation.

## Data Acquisition

### Option 1: Use Existing Dataset (Recommended)

If you have access to the FightIQ dataset:
1. Copy data files to `data/` directory
2. Update `config/config.yaml` with correct paths
3. Run `python scripts/train_production.py`

### Option 2: Scrape Data (Advanced)

Use the provided scraping scripts:
```bash
# Scrape UFC Stats
python scripts/scrape_ufc_stats.py

# Scrape betting odds
python scripts/fetch_odds_bestfightodds.py

# Build golden dataset
python scripts/build_golden_dataset.py
```

### Option 3: Use Test Data

For testing/development only:
```bash
# Generate synthetic test data
python tests/fixtures/test_data.py

# Run with test data
python scripts/train_baseline.py --test-mode
```

## Environment Variables

Create `.env` file:
```bash
# Data paths
DATA_ROOT=/path/to/data

# API keys
ODDS_API_KEY=your_key_here
UFC_API_KEY=your_key_here
```

## Troubleshooting

**Error: Data file not found**
- Check file exists: `ls -la data/`
- Check config: `cat config/config.yaml`
- Use test data: `python tests/fixtures/test_data.py`

**Error: Missing columns**
- Verify schema matches: `python scripts/validate_data.py`
- Update column mappings in config

**Error: Permission denied**
- Check file permissions: `chmod 644 data/*.csv`
```

---

## Fix 5: Add .env Support

Update `src/utils/config.py` to support environment variables:

```python
"""Configuration management with environment variable support"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in config values"""
    if isinstance(value, str):
        # Support ${VAR_NAME:default_value} syntax
        if value.startswith('${') and ':' in value:
            var_name = value[2:value.index(':')]
            default_val = value[value.index(':')+1:-1]
            return os.getenv(var_name, default_val)
        # Support ${VAR_NAME} syntax
        elif value.startswith('${'):
            var_name = value[2:-1]
            return os.getenv(var_name, value)
        return value
    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    return value

def get_config(config_path: str = "config/config.yaml") -> Any:
    """
    Load configuration from YAML file with environment variable expansion
    
    Args:
        config_path: Path to config file
        
    Returns:
        Config object with attributes
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Expand environment variables
    config_dict = expand_env_vars(config_dict)
    
    # Convert dict to object for dot notation access
    return dict_to_obj(config_dict)
```

---

## Implementation Checklist

- [ ] Update `config/config.yaml` with relative paths
- [ ] Add `.env` file support to `src/utils/config.py`
- [ ] Add error handling to `src/data/loaders.py`
- [ ] Create `tests/fixtures/test_data.py` for synthetic data
- [ ] Update tests to use fixtures or skip when data missing
- [ ] Create `DATA_SETUP.md` documentation
- [ ] Create `.env.example` template
- [ ] Update `.gitignore` to exclude `.env`
- [ ] Test on Linux/Mac to verify cross-platform compatibility
- [ ] Run full test suite: `pytest tests/ -v`

## Estimated Time

- Configuration fixes: 2 hours
- Error handling: 2 hours
- Test fixtures: 4 hours
- Documentation: 2 hours
- Testing: 2 hours

**Total**: ~12 hours

---

## After Fixes

Once these fixes are applied:

1. **Test on multiple platforms**:
   ```bash
   # Linux/Mac
   python scripts/train_baseline.py
   pytest tests/ -v
   ```

2. **Verify predictions work**:
   ```bash
   python scripts/predict_upcoming_ufc321.py
   python show_summary.py
   ```

3. **Update documentation**:
   - Mark issues as resolved
   - Update README with new setup instructions
   - Document .env configuration

4. **Deploy to production**:
   - Docker container
   - CI/CD pipeline
   - Monitoring setup

---

**Status**: Ready for implementation  
**Priority**: High  
**Next**: Begin with Fix 1 (Configuration Paths)
