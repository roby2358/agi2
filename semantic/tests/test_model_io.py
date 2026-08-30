"""Tests for shared checkpoint loading."""

import pytest
import torch
from src.config import AGI2Config
from src.model import AGI2Model
from src.model_io import load_model_and_tokenizer
from src.rwkv import RWKVModel


def tiny_config() -> AGI2Config:
    return AGI2Config(
        vocab_size=100,
        n_positions=32,
        n_ctx=32,
        n_embd=32,
        n_layer=2,
        n_head=4,
    )


def save_checkpoint(tmp_path, model, extra=None):
    checkpoint = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "loss": 0.0,
        "config": model.config,
        "model_type": type(model).__name__,
        "tokenizer": {"dummy": True},
    }
    if extra:
        checkpoint.update(extra)
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    return path


@pytest.mark.unit
class TestLoadModelAndTokenizer:
    def test_roundtrip_transformer(self, tmp_path):
        model = AGI2Model(tiny_config())
        path = save_checkpoint(tmp_path, model)
        loaded, tokenizer = load_model_and_tokenizer(path, "cpu", AGI2Model)
        assert isinstance(loaded, AGI2Model)
        assert tokenizer == {"dummy": True}

    def test_roundtrip_rwkv(self, tmp_path):
        model = RWKVModel(tiny_config())
        path = save_checkpoint(tmp_path, model)
        loaded, tokenizer = load_model_and_tokenizer(path, "cpu", RWKVModel)
        assert isinstance(loaded, RWKVModel)

    def test_model_type_mismatch(self, tmp_path):
        model = RWKVModel(tiny_config())
        path = save_checkpoint(tmp_path, model)
        with pytest.raises(ValueError, match="RWKVModel"):
            load_model_and_tokenizer(path, "cpu", AGI2Model)

    def test_missing_state_dict(self, tmp_path):
        path = tmp_path / "bad.pt"
        torch.save({"epoch": 1}, path)
        with pytest.raises(ValueError, match="model_state_dict"):
            load_model_and_tokenizer(path, "cpu", AGI2Model)

    def test_legacy_checkpoint_without_model_type(self, tmp_path):
        # Checkpoints saved before model_type was recorded still load.
        model = AGI2Model(tiny_config())
        path = save_checkpoint(tmp_path, model)
        checkpoint = torch.load(path, weights_only=False)
        del checkpoint["model_type"]
        torch.save(checkpoint, path)
        loaded, _ = load_model_and_tokenizer(path, "cpu", AGI2Model)
        assert isinstance(loaded, AGI2Model)
