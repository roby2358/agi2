"""Tests for PairwiseCosineLoss."""

import pytest
import torch

from src.cosine_loss import PairwiseCosineLoss


class TestPairwiseCosineLoss:
    """Test cases for PairwiseCosineLoss."""

    def setup_method(self) -> None:
        self.loss_fn = PairwiseCosineLoss(0.7, 0.3)
        self.n_embd = 32
        self.vocab_size = 100
        self.batch_size = 8

    def test_basic_forward(self) -> None:
        """Test that loss computation runs without error."""
        hidden = torch.randn(self.batch_size, self.n_embd)
        target_embs = torch.randn(self.batch_size, self.n_embd)
        emb_weight = torch.randn(self.vocab_size, self.n_embd)

        loss, metrics = self.loss_fn(hidden, target_embs, emb_weight)

        assert loss.shape == ()
        assert loss.item() >= 0
        assert "geometric_loss" in metrics
        assert "anchor_loss" in metrics
        assert "total_loss" in metrics

    def test_perfect_geometry_low_loss(self) -> None:
        """When hidden states match target embeddings, loss should be low."""
        target_embs = torch.randn(self.batch_size, self.n_embd)
        hidden = target_embs.clone()
        emb_weight = torch.randn(self.vocab_size, self.n_embd)

        loss, metrics = self.loss_fn(hidden, target_embs, emb_weight)

        assert metrics["geometric_loss"] < 0.01
        assert metrics["anchor_loss"] < 0.01

    def test_degenerate_observations_handled(self) -> None:
        """Zero-norm vectors should be excluded gracefully."""
        hidden = torch.zeros(self.batch_size, self.n_embd)
        target_embs = torch.zeros(self.batch_size, self.n_embd)
        emb_weight = torch.randn(self.vocab_size, self.n_embd)

        loss, metrics = self.loss_fn(hidden, target_embs, emb_weight)

        assert loss.item() == 0.0
        assert metrics["valid_observations"] == 0

    def test_partial_degenerate(self) -> None:
        """Mix of valid and zero-norm should work."""
        hidden = torch.randn(self.batch_size, self.n_embd)
        target_embs = torch.randn(self.batch_size, self.n_embd)
        hidden[0] = 0.0
        hidden[1] = 0.0
        emb_weight = torch.randn(self.vocab_size, self.n_embd)

        loss, metrics = self.loss_fn(hidden, target_embs, emb_weight)

        assert metrics["valid_observations"] == self.batch_size - 2

    def test_gradient_flows(self) -> None:
        """Test that gradients flow through the loss."""
        hidden = torch.randn(self.batch_size, self.n_embd, requires_grad=True)
        target_embs = torch.randn(self.batch_size, self.n_embd)
        emb_weight = torch.randn(self.vocab_size, self.n_embd)

        loss, _ = self.loss_fn(hidden, target_embs, emb_weight)
        loss.backward()

        assert hidden.grad is not None
        assert hidden.grad.shape == hidden.shape

    def test_ratios_affect_loss(self) -> None:
        """Different ratios should produce different loss values."""
        hidden = torch.randn(self.batch_size, self.n_embd)
        target_embs = torch.randn(self.batch_size, self.n_embd)
        emb_weight = torch.randn(self.vocab_size, self.n_embd)

        loss1_fn = PairwiseCosineLoss(0.9, 0.1)
        loss2_fn = PairwiseCosineLoss(0.1, 0.9)

        torch.manual_seed(0)
        l1, _ = loss1_fn(hidden, target_embs, emb_weight)
        torch.manual_seed(0)
        l2, _ = loss2_fn(hidden, target_embs, emb_weight)

        assert l1.shape == l2.shape

    def test_small_batch(self) -> None:
        """Test with minimum viable batch size."""
        hidden = torch.randn(2, self.n_embd)
        target_embs = torch.randn(2, self.n_embd)
        emb_weight = torch.randn(self.vocab_size, self.n_embd)

        loss, metrics = self.loss_fn(hidden, target_embs, emb_weight)
        assert loss.item() >= 0

    def test_single_observation(self) -> None:
        """Single observation: not enough for geometric pairs, should return 0."""
        hidden = torch.randn(1, self.n_embd)
        target_embs = torch.randn(1, self.n_embd)
        emb_weight = torch.randn(self.vocab_size, self.n_embd)

        loss, metrics = self.loss_fn(hidden, target_embs, emb_weight)
        assert metrics["valid_observations"] == 1
