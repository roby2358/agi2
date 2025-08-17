"""Tests for TransformerBlock class."""
import pytest
import torch
from src.transformer import TransformerBlock

class TestTransformerBlock:
    def test_initialization(self):
        block = TransformerBlock(d_model=64, n_heads=8, d_ff=256)
        assert block.d_model == 64
        assert block.n_heads == 8
        assert block.d_ff == 256
    
    def test_forward_pass(self):
        block = TransformerBlock(d_model=64, n_heads=8, d_ff=256)
        x = torch.randn(2, 10, 64)
        output = block(x)
        assert output.shape == (2, 10, 64)
