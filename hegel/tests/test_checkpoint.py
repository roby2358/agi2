"""Tests for src/checkpoint.py."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from src.checkpoint import build_settings, load_weights, save_weights
from src.config import HegelConfig
from src.pipeline import Compiled, compile_from_config

TEXT = "the cat sat on the mat.\nthe dog sat on the log.\n" * 30


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> HegelConfig:
    corpus = tmp_path_factory.mktemp("ckpt") / "corpus.txt"
    corpus.write_text(TEXT)
    return HegelConfig(
        corpus=str(corpus), vocab_size=280, slot_dim=16, logit_dim=24, temperature=0.8
    )


@pytest.fixture(scope="module")
def compiled(config: HegelConfig) -> Compiled:
    return compile_from_config(config)


@pytest.mark.integration
class TestWeightsFile:
    def test_round_trip_gives_the_same_model(
        self, config: HegelConfig, compiled: Compiled, tmp_path: Path
    ) -> None:
        path = tmp_path / "deep" / "weights.pt"
        save_weights(path, compiled.model, compiled.tokenizer, config)
        model, tokenizer, settings = load_weights(path)

        tokens = torch.as_tensor([compiled.tokens[:50]])
        assert torch.equal(model(tokens), compiled.model(tokens))
        text = "the dog sat on the mat."
        assert tokenizer.encode(text) == compiled.tokenizer.encode(text)
        assert tokenizer.decode(tokenizer.encode(text)) == text
        assert settings == build_settings(config)

    def test_rejects_other_files(self, tmp_path: Path) -> None:
        path = tmp_path / "other.pt"
        torch.save({"format": 0}, path)
        with pytest.raises(ValueError):
            load_weights(path)


@pytest.mark.unit
class TestBuildSettings:
    def test_ignores_what_does_not_change_the_weights(
        self, config: HegelConfig
    ) -> None:
        same = replace(
            config, temperature=1.0, seed=7, holdout_fraction=0.1, checkpoint="x.pt"
        )
        assert build_settings(same) == build_settings(config)

    def test_sees_what_does(self, config: HegelConfig) -> None:
        assert build_settings(replace(config, logit_dim=32)) != build_settings(config)
        assert build_settings(replace(config, prune_threshold=1.0)) != build_settings(
            config
        )
