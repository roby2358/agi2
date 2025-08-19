"""Tests for BasicTokenizer class."""
import pytest
from src.basic_tokenizer import BasicTokenizer


class TestBasicTokenizer:
    def test_initialization(self):
        """Test BasicTokenizer initialization."""
        tokenizer = BasicTokenizer()
        assert tokenizer.lowercase == True
        assert tokenizer.vocab == {}
        assert tokenizer.vocab_size == 0
    
    def test_initialization_with_lowercase_false(self):
        """Test BasicTokenizer initialization with lowercase=False."""
        tokenizer = BasicTokenizer(lowercase=False)
        assert tokenizer.lowercase == False
        assert tokenizer.vocab == {}
        assert tokenizer.vocab_size == 0
    
    def test_fit_and_encode_decode(self):
        """Test fitting, encoding, and decoding functionality."""
        tokenizer = BasicTokenizer()
        texts = ["Hello World", "Test Text"]
        tokenizer.fit(texts)
        
        # Test that vocabulary was built
        assert tokenizer.vocab_size > 0
        assert '<PAD>' in tokenizer.vocab
        assert '<UNK>' in tokenizer.vocab
        assert '<BOS>' in tokenizer.vocab
        assert '<EOS>' in tokenizer.vocab
        
        # Test encoding and decoding
        encoded = tokenizer.encode("hello world")
        decoded = tokenizer.decode(encoded)
        assert "hello world" in decoded.lower()
    
    def test_encode_with_unknown_characters(self):
        """Test encoding handles unknown characters correctly."""
        tokenizer = BasicTokenizer()
        texts = ["Hello World"]
        tokenizer.fit(texts)
        
        # Test with unknown character
        encoded = tokenizer.encode("hello world!")
        assert len(encoded) == len("hello world!")
        # Unknown character '!' should map to <UNK> token
        assert tokenizer.vocab['<UNK>'] in encoded
    
    def test_decode_filters_special_tokens(self):
        """Test that decode filters out special tokens."""
        tokenizer = BasicTokenizer()
        texts = ["Hello World"]
        tokenizer.fit(texts)
        
        # Create tokens including special tokens
        special_tokens = [tokenizer.vocab['<PAD>'], tokenizer.vocab['<BOS>'], tokenizer.vocab['<EOS>']]
        regular_tokens = [tokenizer.vocab['h'], tokenizer.vocab['e'], tokenizer.vocab['l'], tokenizer.vocab['l'], tokenizer.vocab['o']]
        
        # Mix special and regular tokens
        mixed_tokens = special_tokens + regular_tokens + special_tokens
        
        decoded = tokenizer.decode(mixed_tokens)
        # Should only contain the regular characters
        assert decoded == "hello"
    
    def test_lowercase_behavior(self):
        """Test that lowercase setting affects encoding."""
        # Test with lowercase=True (default)
        tokenizer_lower = BasicTokenizer(lowercase=True)
        texts = ["Hello World"]
        tokenizer_lower.fit(texts)
        
        encoded_lower = tokenizer_lower.encode("HELLO WORLD")
        decoded_lower = tokenizer_lower.decode(encoded_lower)
        assert decoded_lower == "hello world"
        
        # Test with lowercase=False
        tokenizer_upper = BasicTokenizer(lowercase=False)
        texts_upper = ["HELLO WORLD"]  # Train on uppercase text
        tokenizer_upper.fit(texts_upper)
        
        encoded_upper = tokenizer_upper.encode("HELLO WORLD")
        decoded_upper = tokenizer_upper.decode(encoded_upper)
        assert decoded_upper == "HELLO WORLD"
