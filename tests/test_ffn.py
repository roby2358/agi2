"""Tests for FeedForward class."""
import pytest
import torch
from src.ffn import FeedForward

class TestFeedForward:
    def test_initialization(self):
        ffn = FeedForward(d_model=64, d_ff=256)
        assert ffn.d_model == 64
        assert ffn.d_ff == 256
        assert ffn.activation == "gelu"
    
    def test_forward_pass(self):
        ffn = FeedForward(d_model=64, d_ff=256)
        x = torch.randn(2, 10, 64)
        output = ffn(x)
        assert output.shape == (2, 10, 64)
