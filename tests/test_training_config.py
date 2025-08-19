"""
Test that training script correctly loads TOML configuration values.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config_loader import load_config, get_config_value
from config import AGI2Config


class TestTrainingConfig(unittest.TestCase):
    """Test that TOML configuration values are correctly loaded and used."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config_content = """
# Test configuration
model_name = "test"
model_path = "trained/test.pt"
epochs = 10
batch_size = 8
learning_rate = 1e-4
seq_len = 256
model_positions = 256
model_embd = 384
model_layer = 6
model_head = 6
model_activation = "relu"
model_dropout = 0.2
device = "cpu"
resume = ""
max_length = 100
temperature = 0.8
model_seed = "Test seed text"
max_context_length = 256
sources = ["test_corpus.txt"]
        """
    
    def test_config_loading(self):
        """Test that TOML configuration is correctly loaded."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(self.test_config_content)
            f.flush()
            config_path = f.name
            
            try:
                config = load_config(config_path)
                
                # Test that model architecture parameters are loaded
                self.assertEqual(get_config_value(config, 'model_embd'), 384)
                self.assertEqual(get_config_value(config, 'model_layer'), 6)
                self.assertEqual(get_config_value(config, 'model_head'), 6)
                self.assertEqual(get_config_value(config, 'model_positions'), 256)
                self.assertEqual(get_config_value(config, 'model_activation'), "relu")
                self.assertEqual(get_config_value(config, 'model_dropout'), 0.2)
                
                # Test that other parameters are loaded
                self.assertEqual(get_config_value(config, 'seq_len'), 256)
                self.assertEqual(get_config_value(config, 'epochs'), 10)
                self.assertEqual(get_config_value(config, 'batch_size'), 8)
                
            finally:
                # Windows-friendly file cleanup
                try:
                    os.unlink(config_path)
                except PermissionError:
                    pass  # File will be cleaned up by system later
    
    def test_config_value_fallback(self):
        """Test that get_config_value returns default values when keys don't exist."""
        config = {}
        
        # Test fallback to default values
        self.assertEqual(get_config_value(config, 'model_embd', 768), 768)
        self.assertEqual(get_config_value(config, 'model_layer', 12), 12)
        self.assertEqual(get_config_value(config, 'model_head', 12), 12)
    
    def test_model_config_creation(self):
        """Test that AGI2Config is created with correct values from TOML."""
        config = {
            'model_embd': 384,
            'model_layer': 6,
            'model_head': 6,
            'seq_len': 256,
            'model_activation': 'relu',
            'model_dropout': 0.2
        }
        
        # Simulate what the training script should do
        model_config = AGI2Config(
            vocab_size=1000,
            n_positions=get_config_value(config, 'seq_len', 512),
            n_ctx=get_config_value(config, 'seq_len', 512),
            n_embd=get_config_value(config, 'model_embd', 768),
            n_layer=get_config_value(config, 'model_layer', 12),
            n_head=get_config_value(config, 'model_head', 12),
            activation_function=get_config_value(config, 'model_activation', 'gelu'),
            resid_pdrop=get_config_value(config, 'model_dropout', 0.1),
            embd_pdrop=get_config_value(config, 'model_dropout', 0.1),
            attn_pdrop=get_config_value(config, 'model_dropout', 0.1)
        )
        
        # Verify the config was created correctly
        self.assertEqual(model_config.n_embd, 384)
        self.assertEqual(model_config.n_layer, 6)
        self.assertEqual(model_config.n_head, 6)
        self.assertEqual(model_config.n_positions, 256)
        self.assertEqual(model_config.n_ctx, 256)
        self.assertEqual(model_config.activation_function, "relu")
        self.assertEqual(model_config.resid_pdrop, 0.2)
        self.assertEqual(model_config.embd_pdrop, 0.2)
        self.assertEqual(model_config.attn_pdrop, 0.2)


if __name__ == '__main__':
    unittest.main()
