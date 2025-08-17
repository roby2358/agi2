"""
GPT-2 Model

This module provides the main GPT2Model class that integrates all components.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import GPT2Config
from .embeddings import TokenEmbeddings, PositionEmbeddings
from .transformer import TransformerBlock


class GPT2Model(nn.Module):
    """
    Complete GPT-2 model implementation.
    
    Args:
        config: GPT2Config object containing model parameters
    """
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embeddings = TokenEmbeddings(
            config.vocab_size, 
            config.n_embd, 
            config.embd_pdrop
        )
        self.position_embeddings = PositionEmbeddings(
            config.n_positions, 
            config.n_embd
        )
        
        # Dropout for embeddings
        self.dropout = nn.Dropout(config.embd_pdrop)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                config.n_embd,
                config.n_head,
                config.n_inner,
                config.attn_pdrop,
                config.layer_norm_epsilon
            )
            for _ in range(config.n_layer)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Output projection (can be tied with token embeddings)
        if config.tie_word_embeddings:
            self.output_projection = None
        else:
            self.output_projection = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
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
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the GPT-2 model.
        
        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            return_dict: Whether to return a dictionary (for compatibility)
            
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()
        
        # Create causal mask for autoregressive generation on the same device as input_ids
        causal_mask = self._create_causal_mask(seq_len, input_ids.device)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Ensure attention_mask is on the same device
            attention_mask = attention_mask.to(input_ids.device)
            # Combine causal mask with attention mask
            mask = causal_mask & attention_mask.unsqueeze(1)
        else:
            mask = causal_mask
        
        # Token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(seq_len)
        
        # Combine embeddings
        x = token_embeddings + position_embeddings.unsqueeze(0)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            # Ensure mask is properly shaped for transformer blocks
            if mask is not None:
                # For causal mask, we need to expand it to match the attention scores shape
                if mask.dim() == 2:  # (seq_len, seq_len)
                    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            x = transformer_block(x, mask)
        
        # Final layer normalization
        x = self.ln_f(x)
        
        # Output projection
        if self.output_projection is not None:
            logits = self.output_projection(x)
        else:
            # Weight tying with token embeddings
            logits = torch.matmul(x, self.token_embeddings.embedding.weight.t())
        
        return logits
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return ~mask  # Invert so True means "attend to"
    
    def get_num_params(self) -> int:
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def generate(
        self, 
        prompt: str, 
        max_length: int = 50, 
        temperature: float = 1.0,
        tokenizer = None
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
