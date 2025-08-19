"""Tests for BPETokenizer class."""
import pytest
from src.bpe_tokenizer import BPETokenizer


class TestBPETokenizer:
    def test_initialization(self):
        """Test BPETokenizer initialization."""
        tokenizer = BPETokenizer(vocab_size=1000)
        assert tokenizer.vocab_size == 1000
        assert tokenizer.vocab == {}
        assert tokenizer.reverse_vocab == {}
        assert tokenizer.merges == {}
    
    def test_initialization_default_vocab_size(self):
        """Test BPETokenizer initialization with default vocab_size."""
        tokenizer = BPETokenizer()
        assert tokenizer.vocab_size == 50000
        assert tokenizer.vocab == {}
        assert tokenizer.reverse_vocab == {}
        assert tokenizer.merges == {}
    
    def test_fit_builds_vocabulary(self):
        """Test that fit method builds vocabulary from texts."""
        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["Hello World", "Test Text", "Another Example"]
        tokenizer.fit(texts)
        
        # Should have built vocabulary
        assert len(tokenizer.vocab) > 0
        assert len(tokenizer.reverse_vocab) > 0
        
        # Check that reverse_vocab is inverse of vocab
        for char, idx in tokenizer.vocab.items():
            assert tokenizer.reverse_vocab[idx] == char
    
    def test_encode_decode_roundtrip(self):
        """Test that encode followed by decode returns original text."""
        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["Hello World"]
        tokenizer.fit(texts)
        
        original_text = "Hello World"
        encoded = tokenizer.encode(original_text)
        decoded = tokenizer.decode(encoded)
        
        # Should be able to reconstruct the text
        assert decoded == original_text
    
    def test_encode_with_unknown_characters(self):
        """Test encoding handles unknown characters correctly."""
        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["Hello World"]
        tokenizer.fit(texts)
        
        # Test with unknown character
        encoded = tokenizer.encode("Hello World!")
        assert len(encoded) == len("Hello World!")
        
        # Unknown character '!' should map to default token (0)
        # Find the position of '!' in the original text
        exclamation_pos = "Hello World!".index('!')
        assert encoded[exclamation_pos] == 0
    
    def test_vocabulary_consistency(self):
        """Test that vocabulary and reverse_vocabulary are consistent."""
        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["Hello World", "Test Text"]
        tokenizer.fit(texts)
        
        # Check bidirectional mapping
        for char, idx in tokenizer.vocab.items():
            assert tokenizer.reverse_vocab[idx] == char
        
        for idx, char in tokenizer.reverse_vocab.items():
            assert tokenizer.vocab[char] == idx
        
        # Check sizes match
        assert len(tokenizer.vocab) == len(tokenizer.reverse_vocab)
    
    def test_save_load_vocab_placeholders(self):
        """Test that save_vocab and load_vocab methods exist (placeholders)."""
        tokenizer = BPETokenizer()
        
        # These methods should exist but are placeholders
        assert hasattr(tokenizer, 'save_vocab')
        assert hasattr(tokenizer, 'load_vocab')
        
        # Should not raise errors when called
        tokenizer.save_vocab("test.txt")
        tokenizer.load_vocab("test.txt")
    
    def test_merges_attribute(self):
        """Test that merges attribute is properly initialized."""
        tokenizer = BPETokenizer()
        assert hasattr(tokenizer, 'merges')
        assert tokenizer.merges == {}
        
        # Should remain empty for this simplified implementation
        texts = ["Hello World"]
        tokenizer.fit(texts)
        assert tokenizer.merges == {}
