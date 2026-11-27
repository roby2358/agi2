"""Tests for src/generation.py."""

from typing import Callable, Sequence

import numpy as np
import pytest
from src.config import HegelConfig
from src.generation import respond
from src.pipeline import Compiled, compile_from_config
from src.verify import LogProbs, model_predictor

TEXT = "the cat sat on the mat.\nthe dog sat on the log.<|endoftext|>\n" * 30


@pytest.fixture(scope="module")
def compiled(tmp_path_factory: pytest.TempPathFactory) -> Compiled:
    corpus = tmp_path_factory.mktemp("gen") / "corpus.txt"
    corpus.write_text(TEXT)
    return compile_from_config(HegelConfig(corpus=str(corpus), vocab_size=280))


@pytest.fixture(scope="module")
def predict(compiled: Compiled) -> Callable[[Sequence[int]], LogProbs]:
    return model_predictor(compiled.model)


Predict = Callable[[Sequence[int]], LogProbs]


@pytest.mark.integration
class TestRespond:
    def test_continues_with_corpus_text(
        self, compiled: Compiled, predict: Predict
    ) -> None:
        out = respond(
            compiled.tokenizer, predict, "the cat sat", 6, np.random.default_rng(0)
        )
        assert out
        # With 3 tokens of context this corpus is nearly deterministic.
        assert ("the cat sat" + out)[:18] == "the cat sat on the"

    def test_model_and_chain_agree_on_the_same_draws(
        self, compiled: Compiled, predict: Predict
    ) -> None:
        tokenizer = compiled.tokenizer
        from_model = respond(
            tokenizer, predict, "the dog", 20, np.random.default_rng(3)
        )
        from_chain = respond(
            tokenizer,
            compiled.chain.next_log_probs,
            "the dog",
            20,
            np.random.default_rng(3),
        )
        assert from_model == from_chain

    def test_unseen_source_still_generates(
        self, compiled: Compiled, predict: Predict
    ) -> None:
        rng = np.random.default_rng(1)
        assert respond(compiled.tokenizer, predict, "zebra quartz!", 5, rng)

    def test_empty_source_is_rejected(
        self, compiled: Compiled, predict: Predict
    ) -> None:
        with pytest.raises(ValueError):
            respond(compiled.tokenizer, predict, "  ", 5, np.random.default_rng(1))

    def test_length_is_in_tokens(self, compiled: Compiled, predict: Predict) -> None:
        tokenizer = compiled.tokenizer
        out = respond(tokenizer, predict, "the", 4, np.random.default_rng(2))
        assert len(tokenizer.encode(out)) <= 8
        assert respond(tokenizer, predict, "the", 0, np.random.default_rng(2)) == ""
