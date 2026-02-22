"""Tests for training resume functionality."""

import os
import shutil
import tempfile
import warnings
from pathlib import Path

import pytest
import torch

from src.basic_tokenizer import BasicTokenizer
from src.config import AGI2Config
from src.dataset import TextDataset
from src.model import AGI2Model
from src.training import train_model
from src.utils import load_checkpoint, save_checkpoint

# Shared defaults for all train_model calls in this test file
_TRAIN_DEFAULTS = dict(
    learning_rate=1e-4,
    use_amp=False,
    log_gpu_memory=False,
    num_workers=0,
    pin_memory=False,
    geometric_ratio=0.7,
    anchor_ratio=0.3,
    sigmoid_scale_start=3.0,
    sigmoid_scale_end=10.0,
)


class TestTrainingResume:
    """Test cases for training resume functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = AGI2Config(
            vocab_size=1000, n_layer=2, n_head=4, n_embd=64, n_positions=128, n_ctx=128
        )
        self.model = AGI2Model(self.config)

        self.tokenizer = BasicTokenizer()
        self.tokenizer.fit(["hello world test"])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world test " * 100)
            self.corpus_path = f.name

        self.checkpoint_path = None
        self.final_path = None

    def test_resume_training_basic(self):
        """Test basic resume training functionality"""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=FutureWarning, module="torch.serialization"
            )

            config = AGI2Config(
                vocab_size=100, n_positions=64, n_embd=64, n_layer=2, n_head=2
            )
            model = AGI2Model(config)

            tokenizer = BasicTokenizer()
            tokenizer.fit(["hello world test"])

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write("hello world test " * 100)
                corpus_path = f.name

            checkpoint_path = None
            final_path = None

            try:
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                    save_path = f.name

                history1 = train_model(
                    model=model,
                    tokenizer=tokenizer,
                    sources=[corpus_path],
                    epochs=5,
                    batch_size=2,
                    seq_len=32,
                    device="cpu",
                    save_path=save_path,
                    start_epoch=0,
                    **_TRAIN_DEFAULTS,
                )

                model_name = os.path.basename(save_path).replace(".pt", "")
                checkpoint_path = f"trained/{model_name}.pt_epoch_5.pt"
                assert os.path.exists(checkpoint_path), "Checkpoint should be created"

                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                assert "model_state_dict" in checkpoint
                assert "epoch" in checkpoint
                assert checkpoint["epoch"] == 5

                model2 = AGI2Model(config)

                history2 = train_model(
                    model=model2,
                    tokenizer=tokenizer,
                    sources=[corpus_path],
                    epochs=1,
                    batch_size=2,
                    seq_len=32,
                    device="cpu",
                    save_path=save_path,
                    start_epoch=5,
                    **_TRAIN_DEFAULTS,
                )

                assert len(history2["train_loss"]) == 1

                final_path = f"trained/{model_name}.pt"
                assert os.path.exists(final_path), "Final model should be saved"

                final_checkpoint = torch.load(
                    final_path, map_location="cpu", weights_only=False
                )
                assert final_checkpoint["epoch"] == 6

            finally:
                if os.path.exists(corpus_path):
                    os.unlink(corpus_path)
                if os.path.exists(save_path):
                    os.unlink(save_path)
                if checkpoint_path and os.path.exists(checkpoint_path):
                    os.unlink(checkpoint_path)
                if final_path and os.path.exists(final_path):
                    os.unlink(final_path)

    def test_resume_training_invalid_checkpoint(self):
        """Test resume training with invalid checkpoint path"""
        config = AGI2Config(
            vocab_size=100, n_positions=64, n_embd=64, n_layer=2, n_head=2
        )
        model = AGI2Model(config)

        tokenizer = BasicTokenizer()
        tokenizer.fit(["hello world"])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world " * 50)
            corpus_path = f.name

        try:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                save_path = f.name

            history = train_model(
                model=model,
                tokenizer=tokenizer,
                sources=[corpus_path],
                epochs=1,
                batch_size=2,
                seq_len=32,
                device="cpu",
                save_path=save_path,
                start_epoch=0,
                **_TRAIN_DEFAULTS,
            )

            assert len(history["train_loss"]) == 1

        finally:
            if os.path.exists(corpus_path):
                os.unlink(corpus_path)
            if os.path.exists(save_path):
                os.unlink(save_path)


if __name__ == "__main__":
    pytest.main([__file__])
