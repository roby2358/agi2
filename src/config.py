"""
GPT-2 Configuration Management

This module provides the GPT2Config class for managing model architecture parameters.
"""

from typing import Optional
import torch


class GPT2Config:
    """
    Configuration class for GPT-2 model architecture.
    
    Defaults to GPT-2 Small configuration (12 layers, 12 heads, 768 dimensions).
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_ctx: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: Optional[int] = None,
        activation_function: str = "gelu",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        scale_attn_weights: bool = True,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        tie_word_embeddings: bool = True,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner if n_inner is not None else 4 * n_embd
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
    
    @classmethod
    def from_preset(cls, preset: str) -> "GPT2Config":
        """Create configuration from preset (Small, Medium, Large)."""
        presets = {
            "small": cls(n_layer=12, n_head=12, n_embd=768),
            "medium": cls(n_layer=24, n_head=16, n_embd=1024),
            "large": cls(n_layer=36, n_head=20, n_embd=1280),
        }
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        return presets[preset]
    
    def __repr__(self) -> str:
        return f"GPT2Config(n_layer={self.n_layer}, n_head={self.n_head}, n_embd={self.n_embd})"
