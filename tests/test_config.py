"""Tests for AGI2Config class."""

import pytest
from src.config import AGI2Config


class TestAGI2Config:
    """Test cases for AGI2Config class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AGI2Config()
        
        assert config.vocab_size == 50257
        assert config.n_positions == 1024
        assert config.n_ctx == 1024
        assert config.n_embd == 768
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.n_inner == 3072  # 4 * 768
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
        config = AGI2Config(
            vocab_size=10000,
            n_positions=512,
            n_ctx=512,
            n_embd=256,
            n_layer=6,
            n_head=8,
            n_inner=1024,
            activation_function="relu",
            resid_pdrop=0.2,
            embd_pdrop=0.2,
            attn_pdrop=0.2,
            layer_norm_epsilon=1e-6,
            initializer_range=0.01,
            scale_attn_weights=False,
            use_cache=False,
            bos_token_id=0,
            eos_token_id=1,
            tie_word_embeddings=False
        )
        
        assert config.vocab_size == 10000
        assert config.n_positions == 512
        assert config.n_ctx == 512
        assert config.n_embd == 256
        assert config.n_layer == 6
        assert config.n_head == 8
        assert config.n_inner == 1024
        assert config.activation_function == "relu"
        assert config.resid_pdrop == 0.2
        assert config.embd_pdrop == 0.2
        assert config.attn_pdrop == 0.2
        assert config.layer_norm_epsilon == 1e-6
        assert config.initializer_range == 0.01
        assert config.scale_attn_weights is False
        assert config.use_cache is False
        assert config.bos_token_id == 0
        assert config.eos_token_id == 1
        assert config.tie_word_embeddings is False
    
    def test_from_preset(self):
        """Test preset configurations."""
        small_config = AGI2Config.from_preset("small")
        medium_config = AGI2Config.from_preset("medium")
        large_config = AGI2Config.from_preset("large")
        
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
        """Test invalid preset raises error."""
        with pytest.raises(ValueError):
            AGI2Config.from_preset("invalid")
    
    def test_repr(self):
        """Test string representation."""
        config = AGI2Config(n_layer=6, n_head=8, n_embd=512)
        repr_str = repr(config)
        assert "AGI2Config" in repr_str
        assert "n_layer=6" in repr_str
        assert "n_head=8" in repr_str
        assert "n_embd=512" in repr_str
    
    def test_n_inner_default(self):
        """Test that n_inner defaults to 4 * n_embd when not specified."""
        config = AGI2Config(n_embd=256)
        assert config.n_inner == 1024  # 4 * 256
        
        config = AGI2Config(n_embd=256, n_inner=1024)
        assert config.n_inner == 1024
