"""
Configuration management for Kanazawa 3T system.
"""
import os
from pathlib import Path
from typing import Any, Dict
import yaml
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration container."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self.get(key)
    
    @property
    def raw(self) -> Dict[str, Any]:
        """Get raw configuration dictionary."""
        return self._config


def load_config(config_path: str = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default.yaml
    
    Returns:
        Config object
    """
    if config_path is None:
        # Default to configs/default.yaml
        root_dir = Path(__file__).parent.parent.parent
        config_path = root_dir / "configs" / "default.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Override with environment variables if present
    config_dict = _apply_env_overrides(config_dict)
    
    return Config(config_dict)


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to config."""
    # Example: KEIBAAI_MODEL_PARAMS_LEARNING_RATE=0.1
    prefix = "KEIBAAI_"
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Convert KEIBAAI_MODEL_PARAMS_LEARNING_RATE to model.params.learning_rate
            config_key = key[len(prefix):].lower().replace('_', '.')
            _set_nested_value(config, config_key, value)
    
    return config


def _set_nested_value(config: Dict[str, Any], key: str, value: Any):
    """Set nested dictionary value using dot notation."""
    keys = key.split('.')
    current = config
    
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    
    # Try to convert value to appropriate type
    try:
        if value.lower() in ('true', 'false'):
            value = value.lower() == 'true'
        elif '.' in value:
            value = float(value)
        else:
            value = int(value)
    except (ValueError, AttributeError):
        pass
    
    current[keys[-1]] = value
