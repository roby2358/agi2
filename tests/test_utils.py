"""Tests for utility functions."""
import pytest
import torch
import torch.nn as nn
from src.utils import count_parameters, count_trainable_parameters

class TestUtils:
    def test_count_parameters(self):
        """Test parameter counting functions."""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )
        
        total_params = count_parameters(model)
        trainable_params = count_trainable_parameters(model)
        
        # Linear layers: (10*20 + 20) + (20*5 + 5) = 200 + 20 + 100 + 5 = 325
        assert total_params == 325
        assert trainable_params == 325
