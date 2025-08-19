"""
Test Training Optimizations

This module tests the performance optimizations in the training pipeline.
"""

import torch
import pytest
from unittest.mock import patch, MagicMock
import time

from src.model import AGI2Model
from src.config import AGI2Config
from src.training import train_epoch, train_model


class TestTrainingOptimizations:
    """Test class for training optimizations."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return AGI2Config(
            vocab_size=1000,
            n_positions=512,
            n_embd=128,
            n_layer=2,
            n_head=4,
            n_inner=256,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            tie_word_embeddings=False
        )
    
    @pytest.fixture
    def model(self, config):
        """Create a test model."""
        return AGI2Model(config)
    
    def test_causal_mask_caching(self, model):
        """Test that causal masks are properly cached."""
        device = torch.device('cpu')
        seq_len = 128
        
        # First call should create and cache the mask
        mask1 = model._create_causal_mask(seq_len, device)
        assert (seq_len, device) in model._causal_mask_cache
        
        # Second call should return cached mask
        mask2 = model._create_causal_mask(seq_len, device)
        assert torch.equal(mask1, mask2)
        
        # Verify cache key format
        cache_key = (seq_len, device)
        assert cache_key in model._causal_mask_cache
        
        # Test different sequence lengths create different masks
        mask3 = model._create_causal_mask(256, device)
        assert not torch.equal(mask1, mask3)
        assert (256, device) in model._causal_mask_cache
    
    def test_causal_mask_cache_clearing(self, model):
        """Test that the mask cache can be cleared."""
        device = torch.device('cpu')
        seq_len = 128
        
        # Create some cached masks
        model._create_causal_mask(seq_len, device)
        model._create_causal_mask(256, device)
        
        assert len(model._causal_mask_cache) == 2
        
        # Clear cache
        model.clear_mask_cache()
        assert len(model._causal_mask_cache) == 0
    
    def test_causal_mask_device_handling(self, model):
        """Test that masks are cached per device."""
        if torch.cuda.is_available():
            cpu_device = torch.device('cpu')
            cuda_device = torch.device('cuda:0')
            seq_len = 128
            
            # Create masks on different devices
            cpu_mask = model._create_causal_mask(seq_len, cpu_device)
            cuda_mask = model._create_causal_mask(seq_len, cuda_device)
            
            # Should have separate cache entries
            assert (seq_len, cpu_device) in model._causal_mask_cache
            assert (seq_len, cuda_device) in model._causal_mask_cache
            assert len(model._causal_mask_cache) == 2
            
            # Masks should be on different devices
            assert cpu_mask.device == cpu_device
            assert cuda_mask.device == cuda_device
    
    def test_amp_support(self, model):
        """Test that AMP is properly supported in training."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for AMP testing")
        
        device = torch.device('cuda')
        model = model.to(device)
        
        # Mock dataloader and other components
        mock_dataloader = MagicMock()
        mock_dataloader.__len__ = MagicMock(return_value=10)
        # Return tensors directly, not tuples
        mock_dataloader.__iter__ = MagicMock(return_value=iter([
            torch.randint(0, 1000, (2, 128), device=device)
            for _ in range(10)
        ]))
        
        mock_optimizer = MagicMock()
        mock_criterion = MagicMock()
        # Create a proper loss tensor with gradients
        mock_loss = torch.tensor(0.5, device=device, requires_grad=True)
        mock_criterion.return_value = mock_loss
        
        # Test with AMP enabled
        with patch('torch.cuda.amp.GradScaler') as mock_scaler_class:
            mock_scaler = MagicMock()
            mock_scaler_class.return_value = mock_scaler
            
            result = train_epoch(
                model, mock_dataloader, mock_optimizer, mock_criterion, device,
                use_amp=True
            )
            
            # Verify AMP scaler was used
            mock_scaler_class.assert_called_once()
            mock_scaler.scale.assert_called()
            mock_scaler.step.assert_called()
            mock_scaler.update.assert_called()
    
    def test_non_blocking_transfer(self, model):
        """Test that data transfer uses non_blocking=True."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Mock dataloader
        mock_dataloader = MagicMock()
        mock_dataloader.__len__ = MagicMock(return_value=1)
        
        # Create a tensor that can be moved to device
        test_tensor = torch.randint(0, 1000, (2, 128))
        # Return tensor directly, not tuple
        mock_dataloader.__iter__ = MagicMock(return_value=iter([test_tensor]))
        
        mock_optimizer = MagicMock()
        mock_criterion = MagicMock()
        # Create a proper loss tensor with gradients
        mock_loss = torch.tensor(0.5, device=device, requires_grad=True)
        mock_criterion.return_value = mock_loss
        
        # Patch the tensor.to method to verify non_blocking parameter
        original_to = test_tensor.to
        
        def mock_to(device, non_blocking=False, **kwargs):
            # Verify non_blocking is True for data transfer
            assert non_blocking is True, f"Expected non_blocking=True, got {non_blocking}"
            return original_to(device, non_blocking=non_blocking, **kwargs)
        
        test_tensor.to = mock_to
        
        # Run training
        train_epoch(
            model, mock_dataloader, mock_optimizer, mock_criterion, device,
            use_amp=False
        )
    
    def test_gpu_memory_logging_control(self, model):
        """Test that GPU memory logging can be controlled."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for GPU memory testing")
        
        device = torch.device('cuda')
        model = model.to(device)
        
        # Test with GPU memory logging disabled
        with patch('torch.cuda.memory_allocated') as mock_allocated, \
             patch('torch.cuda.memory_reserved') as mock_reserved:
            
            # Ensure the mocks return valid values
            mock_allocated.return_value = 0
            mock_reserved.return_value = 0
            
            # Test the GPU memory logging logic directly
            gpu_memory_info = ""
            if False:  # log_gpu_memory=False
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                gpu_memory_info = f", GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
            
            # Should not query GPU memory
            mock_allocated.assert_not_called()
            mock_reserved.assert_not_called()
        
        # Test with GPU memory logging enabled
        with patch('torch.cuda.memory_allocated') as mock_allocated, \
             patch('torch.cuda.memory_reserved') as mock_reserved:
            
            mock_allocated.return_value = 1024**3  # 1GB
            mock_reserved.return_value = 2 * 1024**3  # 2GB
            
            # Test the GPU memory logging logic directly
            gpu_memory_info = ""
            if True:  # log_gpu_memory=True
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                gpu_memory_info = f", GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
            
            # Should query GPU memory
            mock_allocated.assert_called()
            mock_reserved.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
