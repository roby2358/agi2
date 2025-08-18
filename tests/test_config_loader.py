"""
Unit tests for the configuration loader module.
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.config_loader import (
    load_config, 
    get_config_value, 
    validate_required_config,
    get_training_config,
    get_generation_config,
    get_interactive_config
)


class TestConfigLoader:
    """Test cases for configuration loading functionality."""
    
    def test_load_config_valid_toml(self):
        """Test loading a valid TOML configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("""
            # Test config
            corpus_path = "test.txt"
            epochs = 5
            learning_rate = 1e-4
            """)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config['corpus_path'] == "test.txt"
            assert config['epochs'] == 5
            assert config['learning_rate'] == 1e-4
        finally:
            os.unlink(temp_path)
    
    def test_load_config_file_not_found(self):
        """Test loading a non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.toml")
    
    def test_load_config_invalid_toml(self):
        """Test loading an invalid TOML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("""
            # Invalid TOML
            corpus_path = "test.txt"
            epochs = 5
            learning_rate = 1e-4
            [invalid_section
            """)
            temp_path = f.name
        
        try:
            with pytest.raises(Exception):  # tomllib.TOMLDecodeError
                load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_get_config_value_with_default(self):
        """Test getting configuration values with default fallbacks."""
        config = {'corpus_path': 'test.txt', 'epochs': 10}
        
        # Existing key
        assert get_config_value(config, 'corpus_path') == 'test.txt'
        assert get_config_value(config, 'epochs') == 10
        
        # Non-existing key with default
        assert get_config_value(config, 'batch_size', 32) == 32
        assert get_config_value(config, 'learning_rate', 1e-4) == 1e-4
        
        # Non-existing key without default
        assert get_config_value(config, 'missing_key') is None
    
    def test_validate_required_config_success(self):
        """Test successful validation of required configuration keys."""
        config = {'corpus_path': 'test.txt', 'model_name': 'test_model'}
        required_keys = ['corpus_path', 'model_name']
        
        # Should not raise an exception
        validate_required_config(config, required_keys)
    
    def test_validate_required_config_missing_keys(self):
        """Test validation failure when required keys are missing."""
        config = {'corpus_path': 'test.txt'}
        required_keys = ['corpus_path', 'model_name', 'epochs']
        
        with pytest.raises(ValueError, match="Missing required configuration keys"):
            validate_required_config(config, required_keys)
    
    def test_get_training_config_success(self):
        """Test successful loading of training configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("""
            corpus_path = "test.txt"
            model_name = "test_model"
            epochs = 5
            batch_size = 16
            """)
            temp_path = f.name
        
        try:
            config = get_training_config(temp_path)
            assert config['corpus_path'] == "test.txt"
            assert config['model_name'] == "test_model"
            assert config['epochs'] == 5
            assert config['batch_size'] == 16
        finally:
            os.unlink(temp_path)
    
    def test_get_training_config_missing_required_keys(self):
        """Test training configuration loading with missing required keys."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("""
            # Missing required keys
            epochs = 5
            batch_size = 16
            """)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Configuration must contain either 'sources' list or 'corpus_path'"):
                get_training_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_get_generation_config_success(self):
        """Test successful loading of generation configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("""
            model_path = "trained/model.pt"
            max_length = 100
            temperature = 0.8
            """)
            temp_path = f.name
        
        try:
            config = get_generation_config(temp_path)
            assert config['model_path'] == "trained/model.pt"
            assert config['max_length'] == 100
            assert config['temperature'] == 0.8
        finally:
            os.unlink(temp_path)
    
    def test_get_generation_config_missing_required_keys(self):
        """Test generation configuration loading with missing required keys."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("""
            # Missing required keys
            max_length = 100
            temperature = 0.8
            """)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Missing required configuration keys"):
                get_generation_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_get_interactive_config_success(self):
        """Test successful loading of interactive configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("""
            model_path = "trained/model.pt"
            max_context_length = 2048
            device = "cuda"
            """)
            temp_path = f.name
        
        try:
            config = get_interactive_config(temp_path)
            assert config['model_path'] == "trained/model.pt"
            assert config['max_context_length'] == 2048
            assert config['device'] == "cuda"
        finally:
            os.unlink(temp_path)
    
    def test_get_interactive_config_missing_required_keys(self):
        """Test interactive configuration loading with missing required keys."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("""
            # Missing required keys
            max_context_length = 2048
            device = "cuda"
            """)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Missing required configuration keys"):
                get_interactive_config(temp_path)
        finally:
            os.unlink(temp_path)
