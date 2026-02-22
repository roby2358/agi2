"""
AGI2 Model

This module provides the main AGI2Model class that integrates all components.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import AGI2Config
from .embeddings import PositionEmbeddings, TokenEmbeddings
from .transformer import TransformerBlock


class AGI2Model(nn.Module):
    """
    Complete AGI2 model implementation.

    Args:
        config: AGI2Config object containing model parameters
    """

    def __init__(self, config: AGI2Config):
        super().__init__()
        self.config = config

        # Cache for causal masks to avoid recreation
        self._causal_mask_cache = {}

        # Token and position embeddings (frozen — static random codebook)
        self.token_embeddings = TokenEmbeddings(
            config.vocab_size, config.n_embd, config.embd_pdrop
        )
        self.token_embeddings.embedding.weight.requires_grad_(False)
        self.position_embeddings = PositionEmbeddings(config.n_positions, config.n_embd)

        # Dropout for embeddings
        self.dropout = nn.Dropout(config.embd_pdrop)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.n_embd,
                    config.n_head,
                    config.n_inner,
                    config.attn_pdrop,
                    config.layer_norm_epsilon,
                )
                for _ in range(config.n_layer)
            ]
        )

        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Output projection (can be tied with token embeddings)
        if config.tie_word_embeddings:
            self.output_projection = None
        else:
            self.output_projection = nn.Linear(
                config.n_embd, config.vocab_size, bias=False
            )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for the model."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _run_transformer(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run input through embeddings, transformer blocks, and final layer norm.

        Returns hidden states of shape (batch_size, seq_len, n_embd).
        """
        batch_size, seq_len = input_ids.size()

        causal_mask = self._create_causal_mask(seq_len, input_ids.device)
        mask = causal_mask

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(seq_len)

        x = token_embeddings + position_embeddings.unsqueeze(0)
        x = self.dropout(x)

        for transformer_block in self.transformer_blocks:
            if mask is not None and mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            x = transformer_block(x, mask)

        return self.ln_f(x)

    def _project_to_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits."""
        if self.output_projection is not None:
            return self.output_projection(hidden_states)
        return torch.matmul(hidden_states, self.token_embeddings.embedding.weight.t())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning logits. Used for generation.

        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_len)

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        hidden_states = self._run_transformer(input_ids)
        return self._project_to_logits(hidden_states)

    def forward_hidden(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both logits and hidden states. Used for training.

        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_len)

        Returns:
            Tuple of (logits, hidden_states) where:
            - logits: (batch_size, seq_len, vocab_size)
            - hidden_states: (batch_size, seq_len, n_embd)
        """
        hidden_states = self._run_transformer(input_ids)
        logits = self._project_to_logits(hidden_states)
        return logits, hidden_states

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation with caching."""
        # Create cache key
        cache_key = (seq_len, device)

        # Return cached mask if available
        if cache_key in self._causal_mask_cache:
            return self._causal_mask_cache[cache_key]

        # Create new mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        mask = ~mask  # Invert so True means "attend to"

        # Cache the mask
        self._causal_mask_cache[cache_key] = mask

        return mask

    def get_num_params(self) -> int:
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def clear_mask_cache(self):
        """Clear the cached causal masks to free memory or when switching devices."""
        self._causal_mask_cache.clear()

    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
        tokenizer=None,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            tokenizer: Tokenizer to use for encoding/decoding

        Returns:
            Generated text string
        """
        if tokenizer is None:
            raise ValueError("Tokenizer is required for text generation")

        # This is a placeholder - actual generation logic will be in generation.py
        # For now, just return the prompt
        return prompt
