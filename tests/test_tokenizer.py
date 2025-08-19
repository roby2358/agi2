"""Tests for tokenizer classes.

This module maintains backward compatibility by importing from the new separate tokenizer modules.
For new tests, consider using test_basic_tokenizer.py and test_bpe_tokenizer.py directly.
"""
import pytest
from src.basic_tokenizer import BasicTokenizer
from src.bpe_tokenizer import BPETokenizer


class TestBasicTokenizer:
    def test_initialization(self):
        """Test BasicTokenizer initialization."""
        tokenizer = BasicTokenizer()
        assert tokenizer.lowercase == True
        assert tokenizer.vocab == {}
    
    def test_fit_and_encode_decode(self):
        """Test fitting, encoding, and decoding functionality."""
        tokenizer = BasicTokenizer()
        texts = ["Hello World", "Test Text"]
        tokenizer.fit(texts)
        
        encoded = tokenizer.encode("hello world")
        decoded = tokenizer.decode(encoded)
        assert "hello world" in decoded.lower()


class TestBPETokenizer:
    def test_initialization(self):
        """Test BPETokenizer initialization."""
        tokenizer = BPETokenizer(vocab_size=1000)
        assert tokenizer.vocab_size == 1000
