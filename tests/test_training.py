"""Tests for training functions."""
import pytest
import torch
from src.training import train_epoch

class TestTraining:
    def test_train_epoch_signature(self):
        """Test that train_epoch function has correct signature."""
        # This is a basic test to ensure the function exists and has correct signature
        assert callable(train_epoch)
        
        # Check that it takes the expected parameters
        import inspect
        sig = inspect.signature(train_epoch)
        params = list(sig.parameters.keys())
        
        expected_params = ['model', 'dataloader', 'optimizer', 'criterion', 'device']
        for param in expected_params:
            assert param in params
