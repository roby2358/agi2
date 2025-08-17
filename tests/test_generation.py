"""Tests for generation functions."""
import pytest
from src.generation import generate_text

class TestGeneration:
    def test_generate_text_signature(self):
        """Test that generate_text function has correct signature."""
        assert callable(generate_text)
        
        # Check that it takes the expected parameters
        import inspect
        sig = inspect.signature(generate_text)
        params = list(sig.parameters.keys())
        
        expected_params = ['model', 'prompt', 'max_length', 'temperature', 'top_k', 'top_p', 'tokenizer', 'device']
        for param in expected_params:
            assert param in params
