"""Tests for GPT2Model class."""
import pytest
import torch
from src.model import GPT2Model
from src.config import GPT2Config

class TestGPT2Model:
    def test_initialization(self):
        config = GPT2Config(n_layer=2, n_head=4, n_embd=64)
        model = GPT2Model(config)
        assert model.config == config
        assert len(model.transformer_blocks) == 2
    
    def test_forward_pass(self):
        config = GPT2Config(n_layer=2, n_head=4, n_embd=64)
        model = GPT2Model(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        output = model(input_ids)
        assert output.shape == (2, 10, config.vocab_size)
