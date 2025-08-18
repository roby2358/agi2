"""
Feed-Forward Network

This module provides the FeedForward class for the AGI2 model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation and dropout.
    
    Args:
        d_model: Dimension of the model
        d_ff: Dimension of the feed-forward network
        dropout: Dropout rate
        activation: Activation function ('gelu' or 'relu')
    """
    
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        
        # Linear layers
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First linear layer with activation
        x = self.fc1(x)
        
        if self.activation == "gelu":
            x = F.gelu(x)
        elif self.activation == "relu":
            x = F.relu(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
        x = self.dropout1(x)
        
        # Second linear layer
        x = self.fc2(x)
        x = self.dropout2(x)
        
        return x
