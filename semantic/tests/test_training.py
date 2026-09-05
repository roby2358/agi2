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
from src.training import (
    _collate_fn,
    _compute_batch_loss,
    _next_token_ids,
    train_epoch,
    train_model,
)


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
            seq_len_start=2,
            seq_len_end=32,
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
            early_stop_patience=20,
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

    def test_next_token_ids_dense_targets(self) -> None:
        """Every real position targets its next token; the last real
        position targets the held-out target; padding is ignorable."""
        batch = _collate_fn(
            [
                {
                    "prompt_ids": torch.tensor([10, 11, 12]),
                    "target_ids": torch.tensor([13]),
                },
                {
                    "prompt_ids": torch.tensor([20, 21]),
                    "target_ids": torch.tensor([22]),
                },
            ]
        )
        next_ids = _next_token_ids(
            batch["prompt_ids"], batch["prompt_mask"], batch["target_ids"]
        )
        # Full-length sample: interior positions shift, last gets the target
        assert next_ids[0].tolist() == [11, 12, 13]
        # Padded sample: last REAL position (index 1) gets the target
        assert next_ids[1, 0] == 21
        assert next_ids[1, 1] == 22
        # Masked selection excludes the padding position entirely
        flat = next_ids[batch["prompt_mask"]]
        assert flat.tolist() == [11, 12, 13, 21, 22]

    def test_dense_loss_sees_one_observation_per_token(self) -> None:
        """dense_targets feeds every real position to the loss; the control
        path feeds one per sample."""
        config = AGI2Config(
            vocab_size=50, n_positions=8, n_ctx=8, n_embd=16, n_layer=1, n_head=2
        )
        model = AGI2Model(config)
        loss_fn = PairwiseCosineLoss(0.7, 0.3, 3.0)
        batch = _collate_fn(
            [
                {
                    "prompt_ids": torch.tensor([1, 2, 3, 4]),
                    "target_ids": torch.tensor([5]),
                },
                {
                    "prompt_ids": torch.tensor([6, 7]),
                    "target_ids": torch.tensor([8]),
                },
            ]
        )
        _, dense_metrics = _compute_batch_loss(
            model,
            batch["prompt_ids"],
            batch["prompt_mask"],
            batch["target_ids"],
            loss_fn,
            dense_targets=True,
        )
        _, sparse_metrics = _compute_batch_loss(
            model,
            batch["prompt_ids"],
            batch["prompt_mask"],
            batch["target_ids"],
            loss_fn,
            dense_targets=False,
        )
        assert dense_metrics["valid_observations"] == 6  # 4 + 2 real positions
        assert sparse_metrics["valid_observations"] == 2  # one per sample

    def test_infonce_objective_through_batch_loss(self) -> None:
        """_compute_batch_loss routes target IDS to an InfoNCELoss and the
        dense path trains one observation per real token."""
        from src.cosine_loss import InfoNCELoss

        config = AGI2Config(
            vocab_size=50, n_positions=8, n_ctx=8, n_embd=16, n_layer=1, n_head=2
        )
        model = AGI2Model(config)
        loss_fn = InfoNCELoss(0.7, 3.0, nce_temperature=0.07)
        batch = _collate_fn(
            [
                {
                    "prompt_ids": torch.tensor([1, 2, 3, 4]),
                    "target_ids": torch.tensor([5]),
                },
                {
                    "prompt_ids": torch.tensor([6, 7]),
                    "target_ids": torch.tensor([8]),
                },
            ]
        )
        loss, metrics = _compute_batch_loss(
            model,
            batch["prompt_ids"],
            batch["prompt_mask"],
            batch["target_ids"],
            loss_fn,
            dense_targets=True,
        )
        assert metrics["valid_observations"] == 6
        assert "perplexity" in metrics
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads)

    def test_dense_loss_backward(self) -> None:
        """The dense path must produce finite gradients end to end."""
        config = AGI2Config(
            vocab_size=50, n_positions=8, n_ctx=8, n_embd=16, n_layer=1, n_head=2
        )
        model = AGI2Model(config)
        loss_fn = PairwiseCosineLoss(0.7, 0.3, 3.0)
        batch = _collate_fn(
            [
                {
                    "prompt_ids": torch.tensor([1, 2, 3, 4]),
                    "target_ids": torch.tensor([5]),
                }
            ]
        )
        loss, _ = _compute_batch_loss(
            model,
            batch["prompt_ids"],
            batch["prompt_mask"],
            batch["target_ids"],
            loss_fn,
            dense_targets=True,
        )
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)

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
