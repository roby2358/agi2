"""Tests for training functionality with pairwise cosine similarity loss."""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from src.basic_tokenizer import BasicTokenizer
from src.config import AGI2Config
from src.cosine_loss import PairwiseCosineLoss
from src.dataset import TextDataset
from src.model import AGI2Model
from src.training import _collate_fn, train_epoch, train_model


class TestTraining:
    """Test cases for training functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.config = AGI2Config(
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=64,
            n_positions=128,
            n_ctx=128,
        )
        self.model = AGI2Model(self.config)
        self.tokenizer = BasicTokenizer()
        self.tokenizer.fit(["test text for vocabulary building with enough tokens"])

        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)

    def _train(self, corpus_path, **overrides):
        """Helper to call train_model with all required params."""
        defaults = dict(
            model=self.model,
            tokenizer=self.tokenizer,
            sources=[corpus_path],
            epochs=1,
            batch_size=2,
            learning_rate=1e-4,
            seq_len=32,
            device="cpu",
            save_path="test_model",
            start_epoch=0,
            use_amp=False,
            log_gpu_memory=False,
            num_workers=0,
            pin_memory=False,
            geometric_ratio=0.7,
            anchor_ratio=0.3,
            sigmoid_scale_start=3.0,
            sigmoid_scale_end=10.0,
        )
        defaults.update(overrides)
        return train_model(**defaults)

    def test_train_epoch_signature(self) -> None:
        """Test that train_epoch function has correct signature."""
        import inspect

        sig = inspect.signature(train_epoch)
        params = list(sig.parameters.keys())

        expected_params = [
            "model",
            "dataloader",
            "optimizer",
            "loss_fn",
            "device",
            "clip_grad_norm",
            "scaler",
            "log_gpu_memory",
        ]
        for param in expected_params:
            assert param in params

    def test_collate_fn(self) -> None:
        """Test custom collate function."""
        batch = [
            {
                "prompt_ids": torch.tensor([1, 2, 3]),
                "target_ids": torch.tensor([4]),
            },
            {
                "prompt_ids": torch.tensor([5, 6]),
                "target_ids": torch.tensor([7]),
            },
        ]
        result = _collate_fn(batch)

        assert "prompt_ids" in result
        assert "prompt_mask" in result
        assert "target_ids" in result
        assert result["prompt_ids"].shape == (2, 3)  # padded to max length
        assert result["prompt_mask"].shape == (2, 3)
        # First item should have full mask
        assert result["prompt_mask"][0].all()
        # Second item should have partial mask
        assert result["prompt_mask"][1, :2].all()
        assert not result["prompt_mask"][1, 2]

    def test_train_model_creates_output(self) -> None:
        """Test that train_model creates trained directory and saves model."""
        corpus_path = "temp_corpus.txt"
        with open(corpus_path, "w") as f:
            f.write("test text for training " * 100)

        try:
            history = self._train(corpus_path)

            assert os.path.exists("trained")
            assert os.path.exists("trained/test_model.pt")
            assert "train_loss" in history
            assert len(history["train_loss"]) == 1
        finally:
            if os.path.exists(corpus_path):
                os.remove(corpus_path)

    def test_history_includes_metrics(self) -> None:
        """Test that training history includes metrics."""
        corpus_path = "temp_corpus.txt"
        with open(corpus_path, "w") as f:
            f.write("test text for training " * 100)

        try:
            history = self._train(corpus_path, epochs=2)

            assert "metrics" in history
            assert len(history["metrics"]) == 2
        finally:
            if os.path.exists(corpus_path):
                os.remove(corpus_path)
