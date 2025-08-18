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


def get_sources_list(config: Dict[str, Any]) -> list[str]:
    """
    Get the list of training data sources from configuration.
    Supports both the new 'sources' list and legacy 'corpus_path'.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of source file paths
        
    Raises:
        ValueError: If neither sources nor corpus_path is found
    """
    # Check for new sources list first
    if 'sources' in config and isinstance(config['sources'], list):
        sources = config['sources']
        if not sources:
            raise ValueError("Sources list cannot be empty")
        return sources
    
    # Fall back to legacy corpus_path
    if 'corpus_path' in config:
        return [config['corpus_path']]
    
    raise ValueError("Configuration must contain either 'sources' list or 'corpus_path'")


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
    
    # Required keys for training - either sources or corpus_path
    required_keys = ['model_name']
    
    # Check that we have either sources or corpus_path
    if 'sources' not in config and 'corpus_path' not in config:
        raise ValueError("Configuration must contain either 'sources' list or 'corpus_path'")
    
    # Validate sources if present
    if 'sources' in config:
        if not isinstance(config['sources'], list) or len(config['sources']) == 0:
            raise ValueError("Sources must be a non-empty list")
    
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
