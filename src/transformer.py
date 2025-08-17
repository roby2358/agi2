"""
Transformer Block

This module provides the TransformerBlock class that combines attention and feed-forward layers.
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .ffn import FeedForward


class TransformerBlock(nn.Module):
    """
    Transformer block combining attention, feed-forward, and layer normalization.
    
    Args:
        d_model: Dimension of the model
        n_heads: Number of attention heads
        d_ff: Dimension of the feed-forward network
        dropout: Dropout rate
        layer_norm_epsilon: Epsilon for layer normalization
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization layers
        self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        residual = x
        x = self.ln1(x)
        x = self.attention(x, x, x, mask)
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward with residual connection and layer norm
        residual = x
        x = self.ln2(x)
        x = self.feed_forward(x)
        x = residual + x
        
        return x
