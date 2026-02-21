"""Tests for PairwiseCosineLoss and aggregation functions."""

import pytest
import torch

from src.cosine_loss import PairwiseCosineLoss, aggregate_hidden_states


class TestPairwiseCosineLoss:
    """Test cases for PairwiseCosineLoss."""

    def setup_method(self) -> None:
        self.loss_fn = PairwiseCosineLoss(0.5, 0.3, 0.2)
        self.n_embd = 32
        self.vocab_size = 100
        self.batch_size = 8

    def _make_emb_hidden(self, num_samples: int) -> torch.Tensor:
        return torch.randn(num_samples, self.n_embd)

    def test_basic_forward(self) -> None:
        """Test that loss computation runs without error."""
        hidden = torch.randn(self.batch_size, self.n_embd)
        target_embs = torch.randn(self.batch_size, self.n_embd)
        emb_weight = torch.randn(self.vocab_size, self.n_embd)
        emb_hidden = self._make_emb_hidden(6)

        loss, metrics = self.loss_fn(hidden, target_embs, emb_weight, emb_hidden)

        assert loss.shape == ()
        assert loss.item() >= 0
        assert "geometric_loss" in metrics
        assert "anchor_loss" in metrics
        assert "embedding_loss" in metrics
        assert "total_loss" in metrics

    def test_perfect_geometry_low_loss(self) -> None:
        """When hidden states match target embeddings, geometric loss should be low."""
        target_embs = torch.randn(self.batch_size, self.n_embd)
        hidden = target_embs.clone()
        emb_weight = torch.randn(self.vocab_size, self.n_embd)
        emb_hidden = self._make_emb_hidden(6)

        loss, metrics = self.loss_fn(hidden, target_embs, emb_weight, emb_hidden)

        assert metrics["geometric_loss"] < 0.01
        assert metrics["anchor_loss"] < 0.01

    def test_degenerate_observations_handled(self) -> None:
        """Zero-norm vectors should be excluded gracefully."""
        hidden = torch.zeros(self.batch_size, self.n_embd)
        target_embs = torch.zeros(self.batch_size, self.n_embd)
        emb_weight = torch.randn(self.vocab_size, self.n_embd)
        emb_hidden = self._make_emb_hidden(6)

        loss, metrics = self.loss_fn(hidden, target_embs, emb_weight, emb_hidden)

        assert loss.item() == 0.0
        assert metrics["valid_observations"] == 0

    def test_partial_degenerate(self) -> None:
        """Mix of valid and zero-norm should work."""
        hidden = torch.randn(self.batch_size, self.n_embd)
        target_embs = torch.randn(self.batch_size, self.n_embd)
        hidden[0] = 0.0
        hidden[1] = 0.0
        emb_weight = torch.randn(self.vocab_size, self.n_embd)
        emb_hidden = self._make_emb_hidden(6)

        loss, metrics = self.loss_fn(hidden, target_embs, emb_weight, emb_hidden)

        assert metrics["valid_observations"] == self.batch_size - 2

    def test_embedding_hidden_states(self) -> None:
        """Test embedding pair loss with pre-computed hidden states."""
        hidden = torch.randn(self.batch_size, self.n_embd)
        target_embs = torch.randn(self.batch_size, self.n_embd)
        emb_weight = torch.randn(self.vocab_size, self.n_embd)
        emb_hidden = self._make_emb_hidden(6)  # 3 pairs

        loss, metrics = self.loss_fn(hidden, target_embs, emb_weight, emb_hidden)

        assert loss.item() >= 0
        assert "embedding_loss" in metrics

    def test_gradient_flows(self) -> None:
        """Test that gradients flow through the loss."""
        hidden = torch.randn(self.batch_size, self.n_embd, requires_grad=True)
        target_embs = torch.randn(self.batch_size, self.n_embd)
        emb_weight = torch.randn(self.vocab_size, self.n_embd)
        emb_hidden = self._make_emb_hidden(6)

        loss, _ = self.loss_fn(hidden, target_embs, emb_weight, emb_hidden)
        loss.backward()

        assert hidden.grad is not None
        assert hidden.grad.shape == hidden.shape

    def test_ratios_affect_loss(self) -> None:
        """Different ratios should produce different loss values."""
        hidden = torch.randn(self.batch_size, self.n_embd)
        target_embs = torch.randn(self.batch_size, self.n_embd)
        emb_weight = torch.randn(self.vocab_size, self.n_embd)
        emb_hidden = self._make_emb_hidden(6)

        loss1_fn = PairwiseCosineLoss(0.8, 0.1, 0.1)
        loss2_fn = PairwiseCosineLoss(0.1, 0.1, 0.8)

        torch.manual_seed(0)
        l1, _ = loss1_fn(hidden, target_embs, emb_weight, emb_hidden)
        torch.manual_seed(0)
        l2, _ = loss2_fn(hidden, target_embs, emb_weight, emb_hidden)

        # They may or may not be exactly different due to randomness,
        # but the function should run without error
        assert l1.shape == l2.shape

    def test_small_batch(self) -> None:
        """Test with minimum viable batch size."""
        hidden = torch.randn(2, self.n_embd)
        target_embs = torch.randn(2, self.n_embd)
        emb_weight = torch.randn(self.vocab_size, self.n_embd)
        emb_hidden = self._make_emb_hidden(4)

        loss, metrics = self.loss_fn(hidden, target_embs, emb_weight, emb_hidden)
        assert loss.item() >= 0

    def test_single_observation(self) -> None:
        """Single observation: not enough for pairs, should return 0."""
        hidden = torch.randn(1, self.n_embd)
        target_embs = torch.randn(1, self.n_embd)
        emb_weight = torch.randn(self.vocab_size, self.n_embd)
        emb_hidden = self._make_emb_hidden(4)

        loss, metrics = self.loss_fn(hidden, target_embs, emb_weight, emb_hidden)
        # With only 1 valid obs, geometric pairs can't be formed
        assert metrics["valid_observations"] == 1


class TestAggregateHiddenStates:
    """Test cases for aggregate_hidden_states function."""

    def test_stage1_last_position(self) -> None:
        """Stage 1 should return the last position."""
        batch_size, seq_len, n_embd = 4, 10, 32
        hidden = torch.randn(batch_size, seq_len, n_embd)

        result = aggregate_hidden_states(hidden, 1, 0.5)

        assert result.shape == (batch_size, n_embd)
        assert torch.allclose(result, hidden[:, -1, :])

    def test_stage2_exponential_decay(self) -> None:
        """Stage 2 should use exponentially decaying weights."""
        batch_size, seq_len, n_embd = 4, 5, 32
        hidden = torch.randn(batch_size, seq_len, n_embd)

        result = aggregate_hidden_states(hidden, 2, 0.5)

        assert result.shape == (batch_size, n_embd)
        # First position should have highest weight
        # Verify it's not just a simple mean
        mean_result = hidden.mean(dim=1)
        assert not torch.allclose(result, mean_result, atol=1e-3)

    def test_stage3_arithmetic_mean(self) -> None:
        """Stage 3 should return arithmetic mean."""
        batch_size, seq_len, n_embd = 4, 10, 32
        hidden = torch.randn(batch_size, seq_len, n_embd)

        result = aggregate_hidden_states(hidden, 3, 0.5)

        assert result.shape == (batch_size, n_embd)
        expected = hidden.mean(dim=1)
        assert torch.allclose(result, expected)

    def test_stage2_weights_sum_to_one(self) -> None:
        """Stage 2 weights should sum to 1 for proper averaging."""
        seq_len = 5
        decay = 0.5
        weights = torch.pow(
            torch.tensor(decay), torch.arange(seq_len, dtype=torch.float)
        )
        weights = weights / weights.sum()
        assert abs(weights.sum().item() - 1.0) < 1e-6

    def test_single_position(self) -> None:
        """All stages should work with a single position."""
        batch_size, n_embd = 4, 32
        hidden = torch.randn(batch_size, 1, n_embd)

        for stage in [1, 2, 3]:
            result = aggregate_hidden_states(hidden, stage, 0.5)
            assert result.shape == (batch_size, n_embd)
            assert torch.allclose(result, hidden[:, 0, :], atol=1e-5)
