"""
Text Dataset

This module provides the TextDataset class for loading and preprocessing text data.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Optional
import os


class TextDataset(Dataset):
    """
    Dataset class for text data loading and preprocessing.
    
    Args:
        corpus_path: Path to the text corpus file
        tokenizer: Tokenizer to use for text processing
        seq_len: Length of sequences to generate
        overlap: Number of tokens to overlap between sequences
    """
    
    def __init__(
        self, 
        corpus_path: str, 
        tokenizer, 
        seq_len: int = 1024,
        overlap: int = 0
    ):
        self.corpus_path = corpus_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.overlap = overlap
        
        # Load and tokenize the corpus
        self.tokens = self._load_corpus()
        self.sequences = self._create_sequences()
    
    def _load_corpus(self) -> List[int]:
        """Load and tokenize the text corpus."""
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")
        
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        return tokens
    
    def _create_sequences(self) -> List[List[int]]:
        """Create training sequences from the tokenized corpus."""
        sequences = []
        step = self.seq_len - self.overlap
        
        for i in range(0, len(self.tokens) - self.seq_len + 1, step):
            sequence = self.tokens[i:i + self.seq_len]
            sequences.append(sequence)
        
        return sequences
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sequence at the specified index.
        
        Args:
            idx: Index of the sequence to retrieve
            
        Returns:
            Tensor containing the token sequence
        """
        sequence = self.sequences[idx]
        return torch.tensor(sequence, dtype=torch.long)
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size from the tokenizer."""
        if hasattr(self.tokenizer, 'vocab_size'):
            return self.tokenizer.vocab_size
        elif hasattr(self.tokenizer, 'vocab'):
            return len(self.tokenizer.vocab)
        else:
            raise AttributeError("Tokenizer doesn't have vocab_size or vocab attribute")
    
    def get_corpus_stats(self) -> dict:
        """Get statistics about the corpus."""
        return {
            'total_tokens': len(self.tokens),
            'total_sequences': len(self.sequences),
            'sequence_length': self.seq_len,
            'vocab_size': self.get_vocab_size(),
            'corpus_path': self.corpus_path
        }
