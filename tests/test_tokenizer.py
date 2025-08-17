"""Tests for tokenizer classes."""
import pytest
from src.tokenizer import BasicTokenizer, BPETokenizer

class TestBasicTokenizer:
    def test_initialization(self):
        tokenizer = BasicTokenizer()
        assert tokenizer.lowercase == True
        assert tokenizer.vocab == {}
    
    def test_fit_and_encode_decode(self):
        tokenizer = BasicTokenizer()
        texts = ["Hello World", "Test Text"]
        tokenizer.fit(texts)
        
        encoded = tokenizer.encode("hello world")
        decoded = tokenizer.decode(encoded)
        assert "hello world" in decoded.lower()

class TestBPETokenizer:
    def test_initialization(self):
        tokenizer = BPETokenizer(vocab_size=1000)
        assert tokenizer.vocab_size == 1000
