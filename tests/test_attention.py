"""Tests for MultiHeadAttention class."""
import pytest
import torch
from src.attention import MultiHeadAttention

class TestMultiHeadAttention:
    def test_initialization(self):
        attention = MultiHeadAttention(d_model=64, n_heads=8)
        assert attention.d_model == 64
        assert attention.n_heads == 8
        assert attention.d_k == 8
    
    def test_invalid_heads(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=64, n_heads=7)  # 64 not divisible by 7
