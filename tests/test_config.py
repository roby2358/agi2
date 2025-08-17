"""
Tests for GPT2Config class.
"""

import pytest
import torch
from src.config import GPT2Config


class TestGPT2Config:
    """Test cases for GPT2Config class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GPT2Config()
        
        assert config.vocab_size == 50257
        assert config.n_positions == 1024
        assert config.n_ctx == 1024
        assert config.n_embd == 768
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.n_inner == 4 * 768  # Should be 4 * n_embd by default
        assert config.activation_function == "gelu"
        assert config.resid_pdrop == 0.1
        assert config.embd_pdrop == 0.1
        assert config.attn_pdrop == 0.1
        assert config.layer_norm_epsilon == 1e-5
        assert config.initializer_range == 0.02
        assert config.scale_attn_weights is True
        assert config.use_cache is True
        assert config.bos_token_id == 50256
        assert config.eos_token_id == 50256
        assert config.tie_word_embeddings is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = GPT2Config(
            vocab_size=1000,
            n_embd=512,
            n_layer=6,
            n_head=8,
            n_inner=2048,
            dropout=0.2
        )
        
        assert config.vocab_size == 1000
        assert config.n_embd == 512
        assert config.n_layer == 6
        assert config.n_head == 8
        assert config.n_inner == 2048
        assert config.resid_pdrop == 0.2
        assert config.embd_pdrop == 0.2
        assert config.attn_pdrop == 0.2
    
    def test_preset_configs(self):
        """Test preset configuration creation."""
        small_config = GPT2Config.from_preset("small")
        medium_config = GPT2Config.from_preset("medium")
        large_config = GPT2Config.from_preset("large")
        
        assert small_config.n_layer == 12
        assert small_config.n_head == 12
        assert small_config.n_embd == 768
        
        assert medium_config.n_layer == 24
        assert medium_config.n_head == 16
        assert medium_config.n_embd == 1024
        
        assert large_config.n_layer == 36
        assert large_config.n_head == 20
        assert large_config.n_embd == 1280
    
    def test_invalid_preset(self):
        """Test that invalid preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            GPT2Config.from_preset("invalid")
    
    def test_repr(self):
        """Test string representation."""
        config = GPT2Config(n_layer=6, n_head=8, n_embd=512)
        repr_str = repr(config)
        
        assert "GPT2Config" in repr_str
        assert "n_layer=6" in repr_str
        assert "n_head=8" in repr_str
        assert "n_embd=512" in repr_str
    
    def test_n_inner_calculation(self):
        """Test that n_inner defaults to 4 * n_embd when not specified."""
        config = GPT2Config(n_embd=256)
        assert config.n_inner == 4 * 256
        
        config = GPT2Config(n_embd=256, n_inner=1024)
        assert config.n_inner == 1024
