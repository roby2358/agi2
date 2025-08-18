"""
Token and Position Embeddings

This module provides TokenEmbeddings and PositionEmbeddings classes for the AGI2 model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TokenEmbeddings(nn.Module):
    """
    Token embeddings layer that converts token IDs to dense vectors.
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of the embedding vectors
        dropout: Dropout rate for embeddings
    """
    
    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embedding weights with normal distribution (mean=0, std=0.02)
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: convert token IDs to embeddings.
        
        Args:
            tokens: Token IDs tensor of shape (batch_size, seq_len)
            
        Returns:
            Embeddings tensor of shape (batch_size, seq_len, d_model)
        """
        embeddings = self.embedding(tokens)
        return self.dropout(embeddings)


class PositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for sequence position encoding.
    
    Args:
        max_seq_len: Maximum sequence length
        d_model: Dimension of the embedding vectors
    """
    
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Create position embeddings
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Forward pass: return position embeddings for given sequence length.
        
        Args:
            seq_len: Length of the sequence
            
        Returns:
            Position embeddings tensor of shape (seq_len, d_model)
        """
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        
        return self.pe[:seq_len]
