"""
Text Dataset

This module provides the TextDataset class for loading and preprocessing text data.
Supports curriculum training stages for pairwise cosine similarity training.
"""

import os
import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Dataset class for text data loading and preprocessing.

    Supports curriculum training stages:
    - Stage 1: (prompt, single next token) pairs
    - Stage 2: (prompt, 2-5 token continuation) pairs
    - Stage 3: (prompt, full response) pairs

    Args:
        sources: List of paths to text corpus files, or single corpus path
        tokenizer: Tokenizer to use for text processing
        seq_len: Maximum sequence length
        stage: Curriculum training stage (1, 2, or 3)
    """

    def __init__(
        self,
        sources: str | list[str],
        tokenizer: object,
        seq_len: int,
        stage: int,
    ):
        # Convert single path to list for consistent handling
        if isinstance(sources, str):
            self.sources = [sources]
        else:
            self.sources = sources

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stage = stage

        # Load and tokenize the corpus
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

            # Tokenize the text from this source
            source_tokens = self.tokenizer.encode(text)
            all_tokens.extend(source_tokens)
            print(f"  Loaded {len(source_tokens)} tokens from {source_path}")

        print(f"Total tokens loaded: {len(all_tokens)}")
        return all_tokens

    def _create_sequences(self) -> List[Dict[str, List[int]]]:
        """Create training sequences based on curriculum stage."""
        if self.stage == 1:
            return self._create_stage1_sequences()
        elif self.stage == 2:
            return self._create_stage2_sequences()
        else:
            return self._create_stage3_sequences()

    def _create_stage1_sequences(self) -> List[Dict[str, List[int]]]:
        """Stage 1: prompt + single next token."""
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

    def _create_stage2_sequences(self) -> List[Dict[str, List[int]]]:
        """Stage 2: prompt + 2-5 token continuation."""
        sequences = []
        max_cont = 5
        max_prompt = min(self.seq_len - max_cont, len(self.tokens) - max_cont)
        if max_prompt < 1:
            return sequences

        step = max(1, max_prompt // 2)
        for i in range(0, len(self.tokens) - 2, step):
            remaining = len(self.tokens) - i
            prompt_len = min(max_prompt, remaining - 2)
            if prompt_len < 1:
                break
            max_available_cont = min(max_cont, remaining - prompt_len)
            if max_available_cont < 2:
                break
            cont_len = random.randint(2, max_available_cont)
            prompt = self.tokens[i : i + prompt_len]
            target = self.tokens[i + prompt_len : i + prompt_len + cont_len]
            sequences.append({"prompt_ids": prompt, "target_ids": target})

        return sequences

    def _create_stage3_sequences(self) -> List[Dict[str, List[int]]]:
        """Stage 3: prompt + full response (original fixed-length behavior)."""
        sequences = []
        for i in range(0, len(self.tokens) - self.seq_len + 1, self.seq_len):
            # Split sequence roughly in half: prompt and response
            full_seq = self.tokens[i : i + self.seq_len]
            split_point = len(full_seq) // 2
            prompt = full_seq[:split_point]
            target = full_seq[split_point:]
            sequences.append({"prompt_ids": prompt, "target_ids": target})

        return sequences

    def set_stage(self, stage: int) -> None:
        """Update the curriculum stage and regenerate sequences."""
        self.stage = stage
        self.sequences = self._create_sequences()

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sequence at the specified index.

        Args:
            idx: Index of the sequence to retrieve

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
            "stage": self.stage,
        }
