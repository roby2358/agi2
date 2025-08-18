"""
Multi-Head Attention

This module provides the MultiHeadAttention class for the AGI2 model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
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
        
        self.dropout_layer = nn.Dropout(dropout)
        
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
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = query.size()
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask is on the same device and properly shaped
            mask = mask.to(query.device)
            # For causal mask, we need to expand it to match the attention scores shape
            if mask.dim() == 2:  # (seq_len, seq_len)
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
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
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return ~mask  # Invert so True means "attend to"
