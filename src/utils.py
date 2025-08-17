"""
Utility Functions

This module provides utility functions and helpers for the GPT-2 implementation.
"""

import torch
import torch.nn as nn
import math
import time
from typing import Dict, List, Optional, Tuple, Union
import json
import os


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate the model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def create_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create a causal mask for autoregressive generation.
    
    Args:
        seq_len: Length of the sequence
        device: Device to create the mask on
        
    Returns:
        Causal mask tensor
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return ~mask  # Invert so True means "attend to"


def create_padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Create padding mask from attention mask.
    
    Args:
        attention_mask: Attention mask tensor
        
    Returns:
        Padding mask tensor
    """
    return attention_mask.unsqueeze(1).unsqueeze(2)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def save_config(config: Dict, filepath: str) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save the configuration
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(filepath: str) -> Dict:
    """
    Load configuration from a JSON file.
    
    Args:
        filepath: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    **kwargs
) -> None:
    """
    Save a model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save the checkpoint
        **kwargs: Additional items to save
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: nn.Module,
    optimizer,
    filepath: str,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Load a model checkpoint.
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        filepath: Path to the checkpoint file
        device: Device to load the checkpoint on
        
    Returns:
        Checkpoint dictionary
    """
    if device is None:
        device = torch.device('cpu')
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device_info() -> Dict[str, str]:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': str(torch.cuda.current_device()) if torch.cuda.is_available() else 'N/A'
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name(0)
        info['device_capability'] = str(torch.cuda.get_device_capability(0))
    
    return info


def print_model_summary(model: nn.Module) -> None:
    """
    Print a summary of the model architecture and parameters.
    
    Args:
        model: PyTorch model
    """
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    model_size = get_model_size_mb(model)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    
    print("\nModel architecture:")
    print(model)
    
    print("=" * 60)


def calculate_attention_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate attention weights and output.
    
    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        mask: Optional attention mask
        
    Returns:
        Tuple of (attention_weights, attention_output)
    """
    d_k = query.size(-1)
    
    # Calculate attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax to get attention weights
    attention_weights = torch.softmax(scores, dim=-1)
    
    # Apply attention to values
    attention_output = torch.matmul(attention_weights, value)
    
    return attention_weights, attention_output
