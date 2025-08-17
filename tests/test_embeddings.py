"""
Tests for TokenEmbeddings and PositionEmbeddings classes.
"""

import pytest
import torch
import torch.nn as nn
from src.embeddings import TokenEmbeddings, PositionEmbeddings


class TestTokenEmbeddings:
    """Test cases for TokenEmbeddings class."""
    
    def test_initialization(self):
        """Test TokenEmbeddings initialization."""
        vocab_size = 1000
        d_model = 512
        dropout = 0.1
        
        embeddings = TokenEmbeddings(vocab_size, d_model, dropout)
        
        assert embeddings.vocab_size == vocab_size
        assert embeddings.d_model == d_model
        assert isinstance(embeddings.embedding, nn.Embedding)
        assert embeddings.embedding.num_embeddings == vocab_size
        assert embeddings.embedding.embedding_dim == d_model
        assert isinstance(embeddings.dropout, nn.Dropout)
    
    def test_weight_initialization(self):
        """Test that weights are initialized with correct distribution."""
        embeddings = TokenEmbeddings(100, 64)
        
        # Check that weights are initialized with normal distribution
        weights = embeddings.embedding.weight
        mean = weights.mean().item()
        std = weights.std().item()
        
        # Allow some tolerance for random initialization
        assert abs(mean) < 0.1
        assert 0.01 < std < 0.03
    
    def test_forward_pass(self):
        """Test forward pass with various input shapes."""
        vocab_size = 100
        d_model = 64
        embeddings = TokenEmbeddings(vocab_size, d_model)
        
        # Test single sequence
        tokens = torch.randint(0, vocab_size, (10,))
        output = embeddings(tokens)
        
        assert output.shape == (10, d_model)
        assert output.dtype == torch.float32
        
        # Test batch of sequences
        tokens = torch.randint(0, vocab_size, (3, 10))
        output = embeddings(tokens)
        
        assert output.shape == (3, 10, d_model)
    
    def test_dropout(self):
        """Test that dropout is applied during training."""
        embeddings = TokenEmbeddings(100, 64, dropout=0.5)
        embeddings.train()
        
        tokens = torch.randint(0, 100, (5,))
        output1 = embeddings(tokens)
        output2 = embeddings(tokens)
        
        # With dropout, outputs should be different
        assert not torch.allclose(output1, output2)
        
        # Test in eval mode (no dropout)
        embeddings.eval()
        output1 = embeddings(tokens)
        output2 = embeddings(tokens)
        
        # Without dropout, outputs should be the same
        assert torch.allclose(output1, output2)


class TestPositionEmbeddings:
    """Test cases for PositionEmbeddings class."""
    
    def test_initialization(self):
        """Test PositionEmbeddings initialization."""
        max_seq_len = 1024
        d_model = 512
        
        embeddings = PositionEmbeddings(max_seq_len, d_model)
        
        assert embeddings.max_seq_len == max_seq_len
        assert embeddings.d_model == d_model
        assert hasattr(embeddings, 'pe')
        assert embeddings.pe.shape == (max_seq_len, d_model)
    
    def test_sinusoidal_encoding(self):
        """Test that position embeddings use sinusoidal encoding."""
        max_seq_len = 100
        d_model = 64
        
        embeddings = PositionEmbeddings(max_seq_len, d_model)
        
        # Check that different positions have different embeddings
        pos_emb_0 = embeddings(10)
        pos_emb_1 = embeddings(20)
        
        assert not torch.allclose(pos_emb_0, pos_emb_1)
        
        # Check that embeddings are normalized (roughly)
        assert torch.allclose(pos_emb_0.norm(), torch.tensor(pos_emb_0.shape[0] ** 0.5), atol=1.0)
    
    def test_forward_pass(self):
        """Test forward pass with various sequence lengths."""
        max_seq_len = 100
        d_model = 64
        
        embeddings = PositionEmbeddings(max_seq_len, d_model)
        
        # Test within max sequence length
        seq_len = 50
        output = embeddings(seq_len)
        
        assert output.shape == (seq_len, d_model)
        assert output.dtype == torch.float32
        
        # Test at max sequence length
        output = embeddings(max_seq_len)
        assert output.shape == (max_seq_len, d_model)
    
    def test_sequence_length_limit(self):
        """Test that sequence length cannot exceed maximum."""
        max_seq_len = 100
        d_model = 64
        
        embeddings = PositionEmbeddings(max_seq_len, d_model)
        
        # Should work at max length
        embeddings(max_seq_len)
        
        # Should raise error beyond max length
        with pytest.raises(ValueError, match="exceeds maximum"):
            embeddings(max_seq_len + 1)
    
    def test_position_uniqueness(self):
        """Test that different positions have unique embeddings."""
        max_seq_len = 50
        d_model = 32
        
        embeddings = PositionEmbeddings(max_seq_len, d_model)
        
        # Get embeddings for different positions
        pos_0 = embeddings(10)
        pos_1 = embeddings(20)
        pos_2 = embeddings(30)
        
        # All should be different
        assert not torch.allclose(pos_0, pos_1)
        assert not torch.allclose(pos_1, pos_2)
        assert not torch.allclose(pos_0, pos_2)
    
    def test_buffer_registration(self):
        """Test that position embeddings are registered as buffers."""
        embeddings = PositionEmbeddings(100, 64)
        
        # Check that pe is a buffer (not a parameter)
        assert 'pe' in embeddings._buffers
        assert 'pe' not in embeddings._parameters
        
        # Check that buffer is on the same device as the module
        device = next(embeddings.parameters()).device if list(embeddings.parameters()) else torch.device('cpu')
        assert embeddings.pe.device == device
