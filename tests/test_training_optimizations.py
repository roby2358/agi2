"""
Test Training Optimizations

This module tests the performance optimizations in the training pipeline.
"""

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.config import AGI2Config
from src.cosine_loss import PairwiseCosineLoss
from src.model import AGI2Model
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
            tie_word_embeddings=False,
        )

    @pytest.fixture
    def model(self, config):
        """Create a test model."""
        return AGI2Model(config)

    def test_causal_mask_caching(self, model):
        """Test that causal masks are properly cached."""
        device = torch.device("cpu")
        seq_len = 128

        mask1 = model._create_causal_mask(seq_len, device)
        assert (seq_len, device) in model._causal_mask_cache

        mask2 = model._create_causal_mask(seq_len, device)
        assert torch.equal(mask1, mask2)

        mask3 = model._create_causal_mask(256, device)
        assert not torch.equal(mask1, mask3)
        assert (256, device) in model._causal_mask_cache

    def test_causal_mask_cache_clearing(self, model):
        """Test that the mask cache can be cleared."""
        device = torch.device("cpu")

        model._create_causal_mask(128, device)
        model._create_causal_mask(256, device)
        assert len(model._causal_mask_cache) == 2

        model.clear_mask_cache()
        assert len(model._causal_mask_cache) == 0

    def test_causal_mask_device_handling(self, model):
        """Test that masks are cached per device."""
        if torch.cuda.is_available():
            cpu_device = torch.device("cpu")
            cuda_device = torch.device("cuda:0")
            seq_len = 128

            cpu_mask = model._create_causal_mask(seq_len, cpu_device)
            cuda_mask = model._create_causal_mask(seq_len, cuda_device)

            assert (seq_len, cpu_device) in model._causal_mask_cache
            assert (seq_len, cuda_device) in model._causal_mask_cache
            assert len(model._causal_mask_cache) == 2

            assert cpu_mask.device == cpu_device
            assert cuda_mask.device == cuda_device

    def test_amp_support(self, model, config):
        """Test that AMP is properly supported in training."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for AMP testing")

        device = torch.device("cuda")
        model = model.to(device)

        mock_dataloader = MagicMock()
        mock_dataloader.__len__ = MagicMock(return_value=2)
        mock_dataloader.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "prompt_ids": torch.randint(0, 1000, (2, 64), device=device),
                        "prompt_mask": torch.ones(
                            2, 64, dtype=torch.bool, device=device
                        ),
                        "target_ids": torch.randint(0, 1000, (2, 1), device=device),
                    }
                    for _ in range(2)
                ]
            )
        )

        mock_optimizer = MagicMock()
        loss_fn = PairwiseCosineLoss(0.7, 0.3)
        mock_scaler = MagicMock()

        result = train_epoch(
            model,
            mock_dataloader,
            mock_optimizer,
            loss_fn,
            device,
            1.0,
            mock_scaler,
            False,
        )

        mock_scaler.scale.assert_called()
        mock_scaler.step.assert_called()
        mock_scaler.update.assert_called()

    def test_non_blocking_transfer(self, model):
        """Test that data transfer uses non_blocking=True."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        prompt_ids = torch.randint(0, 1000, (2, 64))
        target_ids = torch.randint(0, 1000, (2, 1))

        non_blocking_used = []
        original_to = prompt_ids.to

        def mock_to(*args, non_blocking=False, **kwargs):
            non_blocking_used.append(non_blocking)
            return original_to(*args, non_blocking=non_blocking, **kwargs)

        prompt_ids.to = mock_to

        mock_dataloader = MagicMock()
        mock_dataloader.__len__ = MagicMock(return_value=1)
        mock_dataloader.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "prompt_ids": prompt_ids,
                        "prompt_mask": torch.ones(2, 64, dtype=torch.bool),
                        "target_ids": target_ids,
                    }
                ]
            )
        )

        mock_optimizer = MagicMock()
        loss_fn = PairwiseCosineLoss(0.7, 0.3)

        train_epoch(
            model,
            mock_dataloader,
            mock_optimizer,
            loss_fn,
            device,
            1.0,
            None,
            False,
        )

        assert any(
            non_blocking_used
        ), "non_blocking=True should be used for data transfer"

    def test_gpu_memory_logging_control(self, model):
        """Test that GPU memory logging can be controlled."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for GPU memory testing")

        device = torch.device("cuda")
        model = model.to(device)

        with (
            patch("torch.cuda.memory_allocated") as mock_allocated,
            patch("torch.cuda.memory_reserved") as mock_reserved,
        ):
            mock_allocated.return_value = 0
            mock_reserved.return_value = 0

            gpu_memory_info = ""
            if False:  # log_gpu_memory=False
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                gpu_memory_info = (
                    f", GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
                )

            mock_allocated.assert_not_called()
            mock_reserved.assert_not_called()

        with (
            patch("torch.cuda.memory_allocated") as mock_allocated,
            patch("torch.cuda.memory_reserved") as mock_reserved,
        ):
            mock_allocated.return_value = 1024**3
            mock_reserved.return_value = 2 * 1024**3

            gpu_memory_info = ""
            if True:  # log_gpu_memory=True
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                gpu_memory_info = (
                    f", GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
                )

            mock_allocated.assert_called()
            mock_reserved.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
