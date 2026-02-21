"""
Multi-Head Attention

This module provides the MultiHeadAttention class for the AGI2 model.
Uses PyTorch's scaled_dot_product_attention for automatic flash attention
on compatible hardware.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention using F.scaled_dot_product_attention.

    Args:
        d_model: Dimension of the model
        n_heads: Number of attention heads
        dropout: Dropout rate for attention weights
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout

        # Linear projections for query, key, value, and output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Initialize weights
        nn.init.normal_(self.w_q.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_k.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_v.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_o.weight, mean=0.0, std=0.02)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional boolean attention mask where True = attend.
                  Shape (1, 1, seq_len, seq_len) or (seq_len, seq_len).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = query.size()

        # Linear projections and reshape for multi-head attention
        # Shape: (batch, n_heads, seq_len, d_k)
        Q = (
            self.w_q(query)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(key)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.w_v(value)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        # Use SDPA — automatically selects flash attention on compatible hardware.
        # Only apply dropout during training.
        drop_p = self.dropout if self.training else 0.0

        if mask is not None:
            mask = mask.to(query.device)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            # SDPA boolean masks: True = masked out. Our masks: True = attend.
            # Convert to float: 0 where attend, -inf where masked.
            float_mask = torch.zeros_like(mask, dtype=Q.dtype)
            float_mask.masked_fill_(~mask, float("-inf"))
            context = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=float_mask, dropout_p=drop_p
            )
        else:
            context = F.scaled_dot_product_attention(
                Q, K, V, is_causal=True, dropout_p=drop_p
            )

        # Reshape and apply output projection
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        return self.w_o(context)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask for autoregressive generation.

        Args:
            seq_len: Length of the sequence
            device: Device to create the mask on

        Returns:
            Causal mask tensor of shape (seq_len, seq_len)
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return ~mask
