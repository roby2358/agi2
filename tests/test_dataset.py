"""Tests for TextDataset class."""
import pytest
import tempfile
import os
from src.dataset import TextDataset
from src.tokenizer import BasicTokenizer

class TestTextDataset:
    def test_initialization(self):
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("This is a test corpus for testing the dataset.")
            temp_file = f.name
        
        try:
            tokenizer = BasicTokenizer()
            tokenizer.fit(["This is a test"])
            
            dataset = TextDataset(temp_file, tokenizer, seq_len=10)
            assert dataset.corpus_path == temp_file
            assert dataset.tokenizer == tokenizer
            assert dataset.seq_len == 10
        finally:
            os.unlink(temp_file)
