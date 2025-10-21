"""
Configuration loader for UFC Master Pipeline.

Loads YAML configuration and provides easy access to settings.
"""

import yaml
from pathlib import Path
from typing import Any, Dict
from loguru import logger


class Config:
    """
    Configuration manager for UFC Master Pipeline.

    Loads config.yaml and provides dot notation access.

    Example:
        config = Config()
        print(config.paths.data_root)
        print(config.model.xgboost.max_depth)
    """

    def __init__(self, config_path: str = None):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config.yaml (default: config/config.yaml)
        """
        if config_path is None:
            # Default to config/config.yaml relative to project root
            project_root = Path(__file__).parents[2]
            config_path = project_root / "config" / "config.yaml"

        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        logger.info(f"✓ Loaded configuration from {self.config_path}")

    def __getattr__(self, name: str) -> Any:
        """Enable dot notation access to config values"""
        if name.startswith('_'):
            return object.__getattribute__(self, name)

        if name in self._config:
            value = self._config[name]
            # Recursively convert dicts to DotDict for nested access
            if isinstance(value, dict):
                return DotDict(value)
            return value

        raise AttributeError(f"Config has no attribute '{name}'")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value with dot notation.

        Args:
            key: Config key (e.g., 'model.xgboost.max_depth')
            default: Default value if key not found

        Returns:
            Config value or default
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def to_dict(self) -> Dict:
        """Return full config as dictionary"""
        return self._config.copy()


class DotDict:
    """Helper class for dot notation access to nested dicts"""

    def __init__(self, data: Dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return object.__getattribute__(self, name)

        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return DotDict(value)
            return value

        raise AttributeError(f"No attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> Dict:
        return self._data.copy()


# Global config instance
_config = None


def get_config(config_path: str = None) -> Config:
    """
    Get global config instance (singleton pattern).

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


if __name__ == "__main__":
    # Test configuration loader
    config = Config()

    print("\n" + "="*80)
    print("CONFIGURATION TEST")
    print("="*80)

    print(f"\nProject name: {config.project.name}")
    print(f"Project version: {config.project.version}")

    print(f"\nData root: {config.paths.data_root}")
    print(f"Golden dataset: {config.paths.golden_dataset}")

    print(f"\nTrain end date: {config.splits.train_end_date}")
    print(f"Val start date: {config.splits.val_start_date}")
    print(f"Test start date: {config.splits.test_start_date}")

    print(f"\nXGBoost max_depth: {config.model.xgboost.max_depth}")
    print(f"XGBoost learning_rate: {config.model.xgboost.learning_rate}")

    print(f"\nBankroll: £{config.betting.bankroll}")
    print(f"Kelly multiplier: {config.betting.kelly_multiplier}")
    print(f"Min edge: {config.betting.min_edge:.1%}")

    print(f"\nRandom state: {config.random_state}")
    print(f"N jobs: {config.n_jobs}")

    # Test get method
    print(f"\nGet method test:")
    print(f"  model.ensemble.enabled: {config.get('model.ensemble.enabled')}")
    print(f"  nonexistent.key: {config.get('nonexistent.key', 'DEFAULT')}")

    print("\n" + "="*80)
    print("✓ Configuration loaded successfully!")
    print("="*80 + "\n")
