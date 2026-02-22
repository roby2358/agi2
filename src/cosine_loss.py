"""
Pairwise Cosine Similarity Loss

Trains language models using geometric relationship preservation against
the embedding codebook. Two loss terms:
- Geometric: hidden states should preserve embedding similarity
- Anchor: hidden states should stay aligned to the embedding space

Loss: (sigmoid(gap * scale) - 0.5)²
where gap = sim(X', Y') - sim(X, Y)

The sigmoid amplifies the gradient signal in the practical range (gaps of
0.05-0.30) while preserving a free pass near zero and saturating at extremes.
Scale ramps linearly over training to tighten tolerances as the model improves.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseCosineLoss(nn.Module):
    """
    Pairwise cosine similarity loss with sigmoid amplification.

    Two pair types:
    - Geometric: (sigmoid(gap * scale) - 0.5)² where gap = sim(H_i, H_j) - sim(E_i, E_j)
    - Anchor: (sigmoid(gap * scale) - 0.5)² where gap = sim(H_i, E_k) - sim(E_i, E_k)

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
        """Compute sigmoid-amplified squared loss from a similarity gap.

        Maps gap through sigmoid(gap * scale), then squares the deviation
        from 0.5. This amplifies the mid-range gradient signal while
        preserving a free pass at zero.
        """
        return (torch.sigmoid(gap * self.sigmoid_scale) - 0.5) ** 2

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
            target_embeddings: Frozen codebook embeddings (batch_size, n_embd)
            embedding_weight: Frozen vocab embedding matrix (vocab_size, n_embd)

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
