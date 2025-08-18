"""Tests for AGI2Model class."""

import pytest
import torch
from src.model import AGI2Model
from src.config import AGI2Config

class TestAGI2Model:
    def test_model_creation(self):
        config = AGI2Config(n_layer=2, n_head=4, n_embd=64)
        model = AGI2Model(config)
        assert model is not None
    
    def test_forward_pass(self):
        config = AGI2Config(n_layer=2, n_head=4, n_embd=64)
        model = AGI2Model(config)
        
        # Create dummy input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Forward pass
        output = model(input_ids)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert output.shape == expected_shape
        
        # Check output type
        assert output.dtype == torch.float32
