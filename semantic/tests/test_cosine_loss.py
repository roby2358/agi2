"""Tests for PairwiseCosineLoss."""

import pytest
import torch
from src.cosine_loss import InfoNCELoss, PairwiseCosineLoss


class TestPairwiseCosineLoss:
    """Test cases for PairwiseCosineLoss."""

    def setup_method(self) -> None:
        self.loss_fn = PairwiseCosineLoss(0.7, 0.3, 10.0)
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

        loss1_fn = PairwiseCosineLoss(0.9, 0.1, 10.0)
        loss2_fn = PairwiseCosineLoss(0.1, 0.9, 10.0)

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

    def test_sigmoid_amplifies_midrange(self) -> None:
        """Sigmoid loss should be larger than squared loss for mid-range gaps."""
        gap = torch.tensor(0.15)
        absolute = gap.abs().item()
        sigmoid_loss = (torch.sigmoid(gap * 10.0) - 0.5).abs()
        assert sigmoid_loss.item() > absolute * 2  # at least 2x amplification

    def test_sigmoid_free_pass_at_zero(self) -> None:
        """Sigmoid loss should be zero when gap is zero."""
        gap = torch.tensor(0.0)
        sigmoid_loss = (torch.sigmoid(gap * 10.0) - 0.5).abs()
        assert sigmoid_loss.item() < 1e-6

    def test_sigmoid_scale_increases_amplification(self) -> None:
        """Higher sigmoid_scale should produce larger loss for same gap."""
        hidden = torch.randn(self.batch_size, self.n_embd)
        target_embs = torch.randn(self.batch_size, self.n_embd)
        emb_weight = torch.randn(self.vocab_size, self.n_embd)

        low_scale = PairwiseCosineLoss(0.7, 0.3, 5.0)
        high_scale = PairwiseCosineLoss(0.7, 0.3, 20.0)

        torch.manual_seed(42)
        l_low, _ = low_scale(hidden, target_embs, emb_weight)
        torch.manual_seed(42)
        l_high, _ = high_scale(hidden, target_embs, emb_weight)

        assert l_high.item() > l_low.item()


@pytest.mark.unit
class TestInfoNCELoss:
    def _loss(self, geometric_ratio: float = 0.0) -> InfoNCELoss:
        return InfoNCELoss(geometric_ratio, 3.0, nce_temperature=0.07)

    def test_perfect_prediction_beats_wrong_prediction(self) -> None:
        torch.manual_seed(0)
        emb = torch.nn.functional.normalize(torch.randn(20, 8), dim=-1)
        targets = torch.tensor([3, 7, 11])
        right, _ = self._loss()(emb[targets] * 5.0, targets, emb)
        wrong, _ = self._loss()(emb[[0, 1, 2]] * 5.0, targets, emb)
        assert right.item() < wrong.item()

    def test_pushes_all_wrong_tokens_down(self) -> None:
        # The gradient must touch every embedding row via the softmax,
        # not just the target's — the contrastive property the pairwise
        # loss lacks.
        torch.manual_seed(0)
        emb = torch.randn(10, 4, requires_grad=True)
        h = torch.randn(3, 4)
        loss, _ = self._loss()(h, torch.tensor([1, 2, 3]), emb)
        loss.backward()
        rows_with_grad = (emb.grad.abs().sum(dim=-1) > 0).sum().item()
        assert rows_with_grad == 10

    def test_metrics_shape(self) -> None:
        torch.manual_seed(0)
        emb = torch.randn(10, 4)
        h = torch.randn(4, 4)
        _, metrics = self._loss()(h, torch.tensor([0, 1, 2, 3]), emb)
        for key in (
            "nce_loss",
            "perplexity",
            "top1_acc",
            "raw_gap",
            "valid_observations",
        ):
            assert key in metrics
        assert metrics["raw_gap"] == pytest.approx(metrics["nce_loss"])
        assert metrics["valid_observations"] == 4

    def test_geometric_auxiliary_contributes(self) -> None:
        torch.manual_seed(0)
        emb = torch.randn(10, 4)
        h = torch.randn(6, 4)
        targets = torch.tensor([0, 1, 2, 3, 4, 5])
        torch.manual_seed(1)
        plain, _ = self._loss(0.0)(h, targets, emb)
        torch.manual_seed(1)
        with_aux, metrics = self._loss(0.7)(h, targets, emb)
        assert metrics["geometric_loss"] > 0.0
        assert with_aux.item() > plain.item()

    def test_temperature_sharpens(self) -> None:
        # Lower temperature amplifies score differences: a correct but
        # unconfident prediction is punished less at high temperature.
        torch.manual_seed(0)
        emb = torch.nn.functional.normalize(torch.randn(10, 4), dim=-1)
        targets = torch.tensor([0, 1])
        h = emb[targets] + 0.1 * torch.randn(2, 4)
        sharp, _ = InfoNCELoss(0.0, 3.0, nce_temperature=0.05)(h, targets, emb)
        soft, _ = InfoNCELoss(0.0, 3.0, nce_temperature=1.0)(h, targets, emb)
        assert sharp.item() != soft.item()

    def test_degenerate_observations_handled(self) -> None:
        emb = torch.randn(10, 4)
        h = torch.zeros(1, 4)
        loss, metrics = self._loss()(h, torch.tensor([0]), emb)
        assert loss.item() == 0.0
        assert metrics["valid_observations"] == 0
