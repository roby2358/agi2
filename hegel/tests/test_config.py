"""Tests for src/config.py."""

from pathlib import Path

import pytest
from src.config import HegelConfig, load_config


@pytest.mark.unit
class TestHegelConfig:
    def test_defaults(self) -> None:
        config = HegelConfig(corpus="x.txt")
        assert config.vocab_size == 512
        assert config.max_context == 3
        assert config.prune_threshold == 0.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"vocab_size": 1},
            {"max_context": 0},
            {"floor": 0.0},
            {"floor": 1.0},
            {"prune_threshold": -1.0},
            {"holdout_fraction": 1.0},
            {"holdout_fraction": -0.1},
            {"holdout_block": 0},
            {"holdout_seeds": 0},
            {"smoothing": "laplace"},
            {"slot_dim": -1},
            {"logit_dim": -1},
            {"match_threshold": 1.0},
            {"low_weight": 0.0},
            {"temperature": 0.0},
            {"decoder_rounds": -1},
        ],
    )
    def test_rejects_bad_values(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            HegelConfig(corpus="x.txt", **kwargs)  # type: ignore[arg-type]

    def test_load(self, tmp_path: Path) -> None:
        path = tmp_path / "c.toml"
        path.write_text('corpus = "a.txt"\nvocab_size = 300\nmax_context = 2\n')
        config = load_config(path)
        assert (config.corpus, config.vocab_size, config.max_context) == (
            "a.txt",
            300,
            2,
        )

    def test_load_rejects_unknown_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "c.toml"
        path.write_text('corpus = "a.txt"\nlearning_rate = 0.1\n')
        with pytest.raises(ValueError, match="learning_rate"):
            load_config(path)

    def test_shipped_config_loads(self) -> None:
        resources = Path(__file__).resolve().parents[1] / "resources"
        assert load_config(resources / "corpus20.toml").slot_dim == 0
        assert load_config(resources / "corpus20_squeezed.toml").slot_dim == 32
