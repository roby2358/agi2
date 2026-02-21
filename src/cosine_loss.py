"""
Pairwise Cosine Similarity Loss

Trains language models using geometric relationship preservation.
The loss measures how well the model's output vectors preserve the geometric
relationships defined by the embedding matrix.

Loss: (sim(X', Y') - sim(X, Y))²
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseCosineLoss(nn.Module):
    """
    Pairwise cosine similarity loss with three pair types:
    - Geometric: (sim(H_i, H_j) - sim(E_i, E_j))²
    - Anchor: (sim(H_i, E_k) - sim(E_i, E_k))²
    - Embedding: (sim(E_i', E_j') - sim(E_i, E_j))²
    """

    def __init__(
        self,
        geometric_ratio: float,
        anchor_ratio: float,
        embedding_ratio: float,
    ):
        super().__init__()
        self.geometric_ratio = geometric_ratio
        self.anchor_ratio = anchor_ratio
        self.embedding_ratio = embedding_ratio

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
    ) -> torch.Tensor:
        """Compute geometric pair loss: hidden-vs-hidden similarity gap."""
        idx_i, idx_j = self._sample_pairs(h.size(0), num_pairs, device)
        if len(idx_i) == 0:
            return torch.tensor(0.0, device=device)

        sim_h = F.cosine_similarity(h[idx_i], h[idx_j], dim=-1)
        sim_e = F.cosine_similarity(e[idx_i], e[idx_j], dim=-1)
        return ((sim_h - sim_e) ** 2).mean()

    def _anchor_loss(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        embedding_weight: torch.Tensor,
        num_pairs: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute anchor pair loss: hidden-vs-random-vocab similarity gap."""
        valid_batch = h.size(0)
        vocab_size = embedding_weight.size(0)

        obs_idx = torch.randint(0, valid_batch, (num_pairs,), device=device)
        emb_idx = torch.randint(0, vocab_size, (num_pairs,), device=device)
        e_k = embedding_weight[emb_idx]

        sim_h_ek = F.cosine_similarity(h[obs_idx], e_k, dim=-1)
        sim_e_ek = F.cosine_similarity(e[obs_idx], e_k, dim=-1)
        return ((sim_h_ek - sim_e_ek) ** 2).mean()

    def _embedding_loss(
        self,
        embedding_hidden_states: torch.Tensor,
        embedding_weight: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute embedding pair loss from pre-computed hidden states."""
        num_emb = embedding_hidden_states.size(0) // 2
        if num_emb == 0:
            return torch.tensor(0.0, device=device)

        emb_h_i = embedding_hidden_states[:num_emb]
        emb_h_j = embedding_hidden_states[num_emb : 2 * num_emb]

        vocab_size = embedding_weight.size(0)
        emb_idx_i = torch.randint(0, vocab_size, (num_emb,), device=device)
        emb_idx_j = torch.randint(0, vocab_size, (num_emb,), device=device)

        sim_h = F.cosine_similarity(emb_h_i, emb_h_j, dim=-1)
        sim_e = F.cosine_similarity(
            embedding_weight[emb_idx_i], embedding_weight[emb_idx_j], dim=-1
        )
        return ((sim_h - sim_e) ** 2).mean()

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_embeddings: torch.Tensor,
        embedding_weight: torch.Tensor,
        embedding_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict[str, float]]:
        """
        Compute pairwise cosine similarity loss.

        Args:
            hidden_states: Aggregated model outputs (batch_size, n_embd)
            target_embeddings: Aggregated known-good embeddings (batch_size, n_embd)
            embedding_weight: Full vocab embedding matrix (vocab_size, n_embd)
            embedding_hidden_states: Hidden states from forwarding random
                vocab tokens through model (num_samples, n_embd)

        Returns:
            Tuple of (total_loss, metrics_dict)
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
                "embedding_loss": 0.0,
                "total_loss": 0.0,
                "valid_observations": len(valid_indices),
            }

        h = hidden_states[valid_indices]
        e = target_embeddings[valid_indices]
        valid_batch = h.size(0)
        num_pairs = max(1, valid_batch // 2)

        geo_loss = self._geometric_loss(h, e, num_pairs, device)
        anc_loss = self._anchor_loss(h, e, embedding_weight, num_pairs, device)
        emb_loss = self._embedding_loss(
            embedding_hidden_states, embedding_weight, device
        )

        total = (
            self.geometric_ratio * geo_loss
            + self.anchor_ratio * anc_loss
            + self.embedding_ratio * emb_loss
        )

        metrics = {
            "geometric_loss": geo_loss.item(),
            "anchor_loss": anc_loss.item(),
            "embedding_loss": emb_loss.item(),
            "total_loss": total.item(),
            "valid_observations": valid_batch,
        }
        return total, metrics


def aggregate_hidden_states(
    hidden_states: torch.Tensor,
    stage: int,
    position_decay: float,
) -> torch.Tensor:
    """
    Aggregate per-position hidden states into a single observation vector.

    Args:
        hidden_states: (batch_size, seq_len, n_embd)
        stage: Curriculum stage (1, 2, or 3)
        position_decay: Exponential decay factor for stage 2

    Returns:
        Aggregated hidden states (batch_size, n_embd)
    """
    if stage == 1:
        return hidden_states[:, -1, :]

    if stage == 2:
        seq_len = hidden_states.size(1)
        weights = torch.pow(
            torch.tensor(position_decay, device=hidden_states.device),
            torch.arange(
                seq_len,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ),
        )
        weights = weights / weights.sum()
        return (hidden_states * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)

    # Stage 3: arithmetic mean
    return hidden_states.mean(dim=1)
