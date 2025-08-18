"""
Configuration loader for AGI2 scripts.
Loads parameters from TOML configuration files.
"""

import tomllib
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a TOML file.
    
    Args:
        config_path: Path to the TOML configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        tomllib.TOMLDecodeError: If TOML file is malformed
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'rb') as f:
        config = tomllib.load(f)
    
    return config


def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Get a configuration value with a default fallback.
    
    Args:
        config: Configuration dictionary
        key: Configuration key to retrieve
        default: Default value if key doesn't exist
        
    Returns:
        Configuration value or default
    """
    return config.get(key, default)


def validate_required_config(config: Dict[str, Any], required_keys: list[str]) -> None:
    """
    Validate that required configuration keys exist.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required configuration keys
        
    Raises:
        ValueError: If any required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")


def get_training_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate training configuration.
    
    Args:
        config_path: Path to the TOML configuration file
        
    Returns:
        Validated training configuration dictionary
    """
    config = load_config(config_path)
    
    # Required keys for training
    required_keys = ['corpus_path', 'model_name']
    validate_required_config(config, required_keys)
    
    return config


def get_generation_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate generation configuration.
    
    Args:
        config_path: Path to the TOML configuration file
        
    Returns:
        Validated generation configuration dictionary
    """
    config = load_config(config_path)
    
    # Required keys for generation
    required_keys = ['model_path']
    validate_required_config(config, required_keys)
    
    return config


def get_interactive_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate interactive configuration.
    
    Args:
        config_path: Path to the TOML configuration file
        
    Returns:
        Validated interactive configuration dictionary
    """
    config = load_config(config_path)
    
    # Required keys for interactive mode
    required_keys = ['model_path']
    validate_required_config(config, required_keys)
    
    return config
