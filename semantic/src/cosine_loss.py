"""
Pairwise Cosine Similarity Loss

Trains language models using geometric relationship preservation against
the embedding codebook. Two loss terms:
- Geometric: hidden states should preserve embedding similarity
- Anchor: hidden states should stay aligned to the embedding space

Loss: |sigmoid(gap * scale) - 0.5|
where gap = sim(X', Y') - sim(X, Y)

The sigmoid amplifies the gradient signal in the practical range (gaps of
0.05-0.30) while preserving a free pass near zero and saturating at extremes.
Scale ramps linearly over training to tighten tolerances as the model improves.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseCosineLoss(nn.Module):
    """
    Pairwise cosine similarity loss with sigmoid amplification.

    Two pair types:
    - Geometric: |sigmoid(gap * scale) - 0.5| where gap = sim(H_i, H_j) - sim(E_i, E_j)
    - Anchor: |sigmoid(gap * scale) - 0.5| where gap = sim(H_i, E_k) - sim(E_i, E_k)

    Metrics include raw_gap (mean absolute similarity gap before sigmoid)
    for scale-independent progress tracking and early stopping.
    """

    def __init__(
        self,
        geometric_ratio: float,
        anchor_ratio: float,
        sigmoid_scale: float,
    ):
        super().__init__()
        self.geometric_ratio = geometric_ratio
        self.anchor_ratio = anchor_ratio
        self.sigmoid_scale = sigmoid_scale

    def _sigmoid_loss(self, gap: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid-amplified absolute loss from a similarity gap.

        Maps gap through sigmoid(gap * scale), then takes absolute deviation
        from 0.5. This amplifies the mid-range gradient signal while
        preserving a free pass at zero.
        """
        return (torch.sigmoid(gap * self.sigmoid_scale) - 0.5).abs()

    def _sample_pairs(
        self, batch_size: int, num_pairs: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample random pairs of indices from batch, ensuring i != j."""
        num_pairs = min(num_pairs, batch_size * (batch_size - 1) // 2)
        if num_pairs == 0:
            empty = torch.zeros(0, dtype=torch.long, device=device)
            return empty, empty.clone()

        idx_i = torch.randint(0, batch_size, (num_pairs,), device=device)
        idx_j = torch.randint(0, batch_size - 1, (num_pairs,), device=device)
        idx_j = idx_j + (idx_j >= idx_i).long()
        return idx_i, idx_j

    def _geometric_loss(
        self, h: torch.Tensor, e: torch.Tensor, num_pairs: int, device: torch.device
    ) -> Tuple[torch.Tensor, float]:
        """Compute geometric pair loss and raw gap."""
        idx_i, idx_j = self._sample_pairs(h.size(0), num_pairs, device)
        if len(idx_i) == 0:
            return torch.tensor(0.0, device=device), 0.0

        sim_h = F.cosine_similarity(h[idx_i], h[idx_j], dim=-1)
        sim_e = F.cosine_similarity(e[idx_i], e[idx_j], dim=-1)
        gap = sim_h - sim_e
        raw_gap = gap.abs().mean().item()
        return self._sigmoid_loss(gap).mean(), raw_gap

    def _anchor_loss(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        embedding_weight: torch.Tensor,
        num_pairs: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, float]:
        """Compute anchor pair loss and raw gap."""
        valid_batch = h.size(0)
        vocab_size = embedding_weight.size(0)

        obs_idx = torch.randint(0, valid_batch, (num_pairs,), device=device)
        emb_idx = torch.randint(0, vocab_size, (num_pairs,), device=device)
        e_k = embedding_weight[emb_idx]

        sim_h_ek = F.cosine_similarity(h[obs_idx], e_k, dim=-1)
        sim_e_ek = F.cosine_similarity(e[obs_idx], e_k, dim=-1)
        gap = sim_h_ek - sim_e_ek
        raw_gap = gap.abs().mean().item()
        return self._sigmoid_loss(gap).mean(), raw_gap

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_embeddings: torch.Tensor,
        embedding_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict[str, float]]:
        """
        Compute pairwise cosine similarity loss.

        Args:
            hidden_states: Last hidden states from model (batch_size, n_embd)
            target_embeddings: Codebook embeddings of the targets (batch_size, n_embd)
            embedding_weight: Vocab embedding matrix (vocab_size, n_embd)

        Returns:
            Tuple of (total_loss, metrics_dict).
            metrics_dict includes raw_gap: the mean absolute similarity gap
            before sigmoid, for scale-independent early stopping.
        """
        device = hidden_states.device

        # Filter degenerate observations (zero-norm vectors)
        h_norms = hidden_states.norm(dim=-1)
        e_norms = target_embeddings.norm(dim=-1)
        valid_mask = (h_norms > 1e-8) & (e_norms > 1e-8)
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]

        if len(valid_indices) < 2:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {
                "geometric_loss": 0.0,
                "anchor_loss": 0.0,
                "total_loss": 0.0,
                "raw_gap": 0.0,
                "valid_observations": len(valid_indices),
            }

        h = hidden_states[valid_indices]
        e = target_embeddings[valid_indices]
        valid_batch = h.size(0)
        num_pairs = max(1, valid_batch // 2)

        geo_loss, geo_gap = self._geometric_loss(h, e, num_pairs, device)
        anc_loss, anc_gap = self._anchor_loss(h, e, embedding_weight, num_pairs, device)

        total = self.geometric_ratio * geo_loss + self.anchor_ratio * anc_loss

        # Weighted raw gap matches the loss weighting
        raw_gap = self.geometric_ratio * geo_gap + self.anchor_ratio * anc_gap

        metrics = {
            "geometric_loss": geo_loss.item(),
            "anchor_loss": anc_loss.item(),
            "total_loss": total.item(),
            "raw_gap": raw_gap,
            "valid_observations": valid_batch,
        }
        return total, metrics


class CrossEntropyLoss(PairwiseCosineLoss):
    """
    Plain cross-entropy over tied-projection logits — the standard-LM
    control for the cosine-training experiment.

    Logits are the unnormalized dot products between hidden states and the
    embedding matrix — exactly what model.forward computes for a
    tie_word_embeddings model — with no temperature and no auxiliary
    terms. Everything else (dense targets, data, curriculum) is shared
    with the cosine objectives, so any quality gap measures the objective
    alone.

    Metrics mirror InfoNCELoss (ce_loss, perplexity, top1_acc) with
    raw_gap aliased to ce_loss for the early-stop machinery.
    """

    def __init__(self) -> None:
        super().__init__(0.0, 0.0, 1.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
        embedding_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict[str, float]]:
        """
        Compute cross-entropy over dot-product logits.

        Args:
            hidden_states: Hidden vectors to train (num_observations, n_embd)
            target_ids: True next-token ids (num_observations,)
            embedding_weight: Vocab embedding matrix (vocab_size, n_embd)
        """
        device = hidden_states.device
        if hidden_states.size(0) < 1:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {
                "ce_loss": 0.0,
                "total_loss": 0.0,
                "perplexity": 0.0,
                "top1_acc": 0.0,
                "raw_gap": 0.0,
                "valid_observations": 0,
            }

        logits = hidden_states @ embedding_weight.t()
        ce = F.cross_entropy(logits, target_ids)

        with torch.no_grad():
            top1 = (logits.argmax(dim=-1) == target_ids).float().mean().item()

        metrics = {
            "ce_loss": ce.item(),
            "total_loss": ce.item(),
            "perplexity": float(torch.exp(ce.detach()).item()),
            "top1_acc": top1,
            "raw_gap": ce.item(),
            "valid_observations": hidden_states.size(0),
        }
        return ce, metrics


class InfoNCELoss(PairwiseCosineLoss):
    """
    Cross-entropy over cosine logits (InfoNCE) with an optional geometric
    auxiliary term.

    This is cosine-similarity training with proper contrastive
    normalization: every vocab embedding is scored by cosine against the
    hidden state, scaled by a temperature, and the true next token's
    softmax probability is maximized. Unlike the pairwise sigmoid-gap loss,
    raising the target necessarily pushes ALL wrong tokens down — the
    normalization the spec's "Loss of Calibrated Confidence" section
    anticipated losing. It also optimizes exactly the distribution
    generation samples from (generation scores by cosine-softmax already).

    The geometric term (structure preservation between hidden states and
    embedding space) is retained as a weighted auxiliary; the anchor term
    is subsumed — InfoNCE anchors against the whole codebook every step.

    Args:
        geometric_ratio: Weight of the auxiliary geometric pair loss
            (0.0 disables it)
        sigmoid_scale: Sigmoid amplification for the geometric term (ramped
            by the training loop exactly as for PairwiseCosineLoss)
        nce_temperature: Divisor for cosine logits. Cosine spans [-1, 1],
            so ~0.07 gives logits in roughly [-14, 14] over the vocab.
    """

    def __init__(
        self,
        geometric_ratio: float,
        sigmoid_scale: float,
        nce_temperature: float = 0.07,
    ):
        super().__init__(geometric_ratio, 0.0, sigmoid_scale)
        self.nce_temperature = nce_temperature

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
        embedding_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict[str, float]]:
        """
        Compute InfoNCE loss over the full vocabulary.

        Args:
            hidden_states: Hidden vectors to train (num_observations, n_embd)
            target_ids: True next-token ids (num_observations,)
            embedding_weight: Vocab embedding matrix (vocab_size, n_embd)

        Returns:
            Tuple of (total_loss, metrics_dict). raw_gap in the metrics is
            the NCE component — temperature is fixed, so it is the
            scale-independent progress measure the early stop reads.
        """
        device = hidden_states.device

        h_norms = hidden_states.norm(dim=-1)
        valid_mask = h_norms > 1e-8
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        if len(valid_indices) < 2:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {
                "nce_loss": 0.0,
                "geometric_loss": 0.0,
                "total_loss": 0.0,
                "perplexity": 0.0,
                "top1_acc": 0.0,
                "raw_gap": 0.0,
                "valid_observations": len(valid_indices),
            }

        h = hidden_states[valid_indices]
        targets = target_ids[valid_indices]
        valid_batch = h.size(0)

        logits = (
            F.normalize(h, dim=-1) @ F.normalize(embedding_weight, dim=-1).t()
        ) / self.nce_temperature
        nce = F.cross_entropy(logits, targets)

        geo_loss: Optional[torch.Tensor] = None
        if self.geometric_ratio > 0.0:
            e = embedding_weight[targets]
            num_pairs = max(1, valid_batch // 2)
            geo_loss, _ = self._geometric_loss(h, e, num_pairs, device)
            total = nce + self.geometric_ratio * geo_loss
        else:
            total = nce

        with torch.no_grad():
            top1 = (logits.argmax(dim=-1) == targets).float().mean().item()

        metrics = {
            "nce_loss": nce.item(),
            "geometric_loss": geo_loss.item() if geo_loss is not None else 0.0,
            "total_loss": total.item(),
            "perplexity": float(torch.exp(nce.detach()).item()),
            "top1_acc": top1,
            "raw_gap": nce.item(),
            "valid_observations": valid_batch,
        }
        return total, metrics
