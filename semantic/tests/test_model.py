"""Tests for AGI2Model class."""

import pytest
import torch

from src.config import AGI2Config
from src.model import AGI2Model


class TestAGI2Model:
    def test_model_creation(self) -> None:
        config = AGI2Config(n_layer=2, n_head=4, n_embd=64)
        model = AGI2Model(config)
        assert model is not None

    def test_forward_pass(self) -> None:
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

    def test_forward_hidden(self) -> None:
        """forward_hidden returns both logits and hidden states."""
        config = AGI2Config(n_layer=2, n_head=4, n_embd=64)
        model = AGI2Model(config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, hidden_states = model.forward_hidden(input_ids)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert hidden_states.shape == (batch_size, seq_len, config.n_embd)

    def test_hidden_states_before_projection(self) -> None:
        """Hidden states should be the pre-projection representation."""
        config = AGI2Config(n_layer=2, n_head=4, n_embd=64, vocab_size=100)
        model = AGI2Model(config)

        batch_size, seq_len = 2, 5
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, hidden_states = model.forward_hidden(input_ids)

        # Hidden state dimension should be n_embd, not vocab_size
        assert hidden_states.size(-1) == config.n_embd
        assert logits.size(-1) == config.vocab_size

    def test_forward_returns_tensor(self) -> None:
        """forward() should return a plain tensor, not a tuple."""
        config = AGI2Config(n_layer=2, n_head=4, n_embd=64)
        model = AGI2Model(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        output = model(input_ids)

        assert isinstance(output, torch.Tensor)
        assert output.dim() == 3

    def test_embedding_weight_accessible(self) -> None:
        """Token embedding weights should be accessible for similarity computation."""
        config = AGI2Config(n_layer=2, n_head=4, n_embd=64, vocab_size=100)
        model = AGI2Model(config)

        emb_weight = model.token_embeddings.embedding.weight
        assert emb_weight.shape == (config.vocab_size, config.n_embd)
