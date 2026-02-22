"""
Text Dataset

This module provides the TextDataset class for loading and preprocessing text data.
Produces (prompt, single next token) pairs for pairwise cosine similarity training.
"""

import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Dataset class for text data loading and preprocessing.

    Produces (prompt, single next token) pairs. The training loop compares the
    last hidden vector against the target token's embedding.

    Args:
        sources: List of paths to text corpus files, or single corpus path
        tokenizer: Tokenizer to use for text processing
        seq_len: Maximum sequence length
    """

    def __init__(
        self,
        sources: str | list[str],
        tokenizer: object,
        seq_len: int,
    ):
        if isinstance(sources, str):
            self.sources = [sources]
        else:
            self.sources = sources

        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.tokens = self._load_corpus()
        self.sequences = self._create_sequences()

    def _load_corpus(self) -> List[int]:
        """Load and tokenize text from multiple sources."""
        all_tokens: List[int] = []

        for source_path in self.sources:
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Source file not found: {source_path}")

            print(f"Loading source: {source_path}")
            with open(source_path, "r", encoding="utf-8") as f:
                text = f.read()

            source_tokens = self.tokenizer.encode(text)
            all_tokens.extend(source_tokens)
            print(f"  Loaded {len(source_tokens)} tokens from {source_path}")

        print(f"Total tokens loaded: {len(all_tokens)}")
        return all_tokens

    def _create_sequences(self) -> List[Dict[str, List[int]]]:
        """Create (prompt, single next token) pairs."""
        sequences = []
        max_prompt = min(self.seq_len - 1, len(self.tokens) - 1)
        if max_prompt < 1:
            return sequences

        step = max(1, max_prompt // 2)
        for i in range(0, len(self.tokens) - 1, step):
            end = min(i + max_prompt, len(self.tokens) - 1)
            prompt_len = end - i
            if prompt_len < 1:
                break
            prompt = self.tokens[i : i + prompt_len]
            target = self.tokens[i + prompt_len : i + prompt_len + 1]
            sequences.append({"prompt_ids": prompt, "target_ids": target})

        return sequences

    def set_seq_len(self, seq_len: int) -> None:
        """Update the sequence length and regenerate sequences."""
        self.seq_len = seq_len
        self.sequences = self._create_sequences()

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sequence at the specified index.

        Returns:
            Dict with 'prompt_ids' and 'target_ids' tensors
        """
        seq = self.sequences[idx]
        return {
            "prompt_ids": torch.tensor(seq["prompt_ids"], dtype=torch.long),
            "target_ids": torch.tensor(seq["target_ids"], dtype=torch.long),
        }

    def get_vocab_size(self) -> int:
        """Get the vocabulary size from the tokenizer."""
        if hasattr(self.tokenizer, "vocab_size"):
            return self.tokenizer.vocab_size
        elif hasattr(self.tokenizer, "vocab"):
            return len(self.tokenizer.vocab)
        else:
            raise AttributeError("Tokenizer doesn't have vocab_size or vocab attribute")

    def get_corpus_stats(self) -> dict:
        """Get statistics about the corpus."""
        return {
            "total_tokens": len(self.tokens),
            "total_sequences": len(self.sequences),
            "sequence_length": self.seq_len,
            "vocab_size": self.get_vocab_size(),
            "sources": self.sources,
            "num_sources": len(self.sources),
        }
