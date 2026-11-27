"""Tests for src/holdout.py."""

from pathlib import Path

import numpy as np
import pytest
from src.config import HegelConfig
from src.holdout import score_seeds


def _corpus(tmp_path: Path) -> str:
    rng = np.random.default_rng(0)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    picks = [words[int(i)] for i in rng.integers(len(words), size=1500)]
    path = tmp_path / "corpus.txt"
    path.write_text(" ".join(picks))
    return str(path)


@pytest.mark.integration
class TestScoreSeeds:
    def test_one_score_per_seed(self, tmp_path: Path) -> None:
        config = HegelConfig(
            corpus=_corpus(tmp_path),
            vocab_size=270,
            holdout_fraction=0.2,
            holdout_block=200,
            holdout_seed=3,
            holdout_seeds=3,
        )
        scores = score_seeds(config)
        assert [s.seed for s in scores] == [3, 4, 5]
        assert all(s.tokens > 0 and s.contexts > 0 for s in scores)
        assert len({s.tokens for s in scores}) > 1

    def test_smoothing_helps_on_unseen_text(self, tmp_path: Path) -> None:
        base = dict(
            corpus=_corpus(tmp_path),
            vocab_size=270,
            holdout_fraction=0.2,
            holdout_block=200,
        )
        hard = score_seeds(HegelConfig(**base))[0]  # type: ignore[arg-type]
        soft = score_seeds(
            HegelConfig(smoothing="witten_bell", **base)  # type: ignore[arg-type]
        )[0]
        assert soft.unigram_nll == pytest.approx(hard.unigram_nll)
        assert soft.unpruned_nll < hard.unpruned_nll
        assert 0.0 <= soft.chain_accuracy <= 1.0
