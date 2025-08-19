"""
Basic Tokenizer

This module provides BasicTokenizer class for simple text preprocessing.
"""

import re
from typing import List, Dict, Set, Tuple


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
