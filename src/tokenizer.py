"""
Tokenization

This module provides BasicTokenizer and BPETokenizer classes for text processing.
"""

import re
from typing import List, Dict, Set, Tuple
from collections import Counter


class BasicTokenizer:
    """
    Basic tokenizer for simple text preprocessing.
    
    Handles basic text preprocessing like lowercase conversion and whitespace handling.
    """
    
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0
    
    def fit(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of text strings to build vocabulary from
        """
        # Simple character-level vocabulary
        chars = set()
        for text in texts:
            if self.lowercase:
                text = text.lower()
            chars.update(text)
        
        # Create vocabulary with special tokens
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.vocab = {token: i for i, token in enumerate(special_tokens)}
        self.vocab.update({char: i + len(special_tokens) for i, char in enumerate(sorted(chars))})
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        if self.lowercase:
            text = text.lower()
        
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab['<UNK>'])
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        text = ""
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                char = self.reverse_vocab[token_id]
                if char not in ['<PAD>', '<BOS>', '<EOS>']:
                    text += char
        
        return text


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer for more sophisticated text processing.
    
    Args:
        vocab_size: Maximum vocabulary size
    """
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}
        self.merges = {}
    
    def fit(self, texts: List[str]) -> None:
        """
        Train BPE tokenizer on a list of texts.
        
        Args:
            texts: List of text strings to train on
        """
        # This is a simplified BPE implementation
        # In practice, you'd want to use a more robust implementation
        
        # Start with character-level vocabulary
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Initialize vocabulary
        self.vocab = {char: i for i, char in enumerate(sorted(chars))}
        
        # Simple merge strategy (this is a placeholder)
        # Real BPE would count bigram frequencies and merge most common pairs
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text using BPE.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        # Simplified encoding - just character-level for now
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                # Handle unknown characters
                tokens.append(0)  # Default to first token
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode BPE token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        text = ""
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                text += self.reverse_vocab[token_id]
        
        return text
    
    def save_vocab(self, filepath: str) -> None:
        """Save vocabulary to file."""
        # Placeholder for vocabulary persistence
        pass
    
    def load_vocab(self, filepath: str) -> None:
        """Load vocabulary from file."""
        # Placeholder for vocabulary loading
        pass
