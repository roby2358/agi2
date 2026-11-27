"""Tests for src/generation.py."""

import numpy as np
import pytest
from src.config import HegelConfig
from src.generation import respond
from src.pipeline import Compiled, compile_from_config

TEXT = "the cat sat on the mat.\nthe dog sat on the log.<|endoftext|>\n" * 30


@pytest.fixture(scope="module")
def compiled(tmp_path_factory: pytest.TempPathFactory) -> Compiled:
    corpus = tmp_path_factory.mktemp("gen") / "corpus.txt"
    corpus.write_text(TEXT)
    return compile_from_config(HegelConfig(corpus=str(corpus), vocab_size=280))


@pytest.mark.integration
class TestRespond:
    def test_continues_with_corpus_text(self, compiled: Compiled) -> None:
        out = respond(compiled, "the cat sat", 6, np.random.default_rng(0))
        assert out
        # With 3 tokens of context this corpus is nearly deterministic.
        assert ("the cat sat" + out)[:18] == "the cat sat on the"

    def test_model_and_chain_agree_on_the_same_draws(self, compiled: Compiled) -> None:
        from_model = respond(compiled, "the dog", 20, np.random.default_rng(3))
        from_chain = respond(
            compiled, "the dog", 20, np.random.default_rng(3), use_chain=True
        )
        assert from_model == from_chain

    def test_unseen_source_still_generates(self, compiled: Compiled) -> None:
        assert respond(compiled, "zebra quartz!", 5, np.random.default_rng(1))

    def test_empty_source_is_rejected(self, compiled: Compiled) -> None:
        with pytest.raises(ValueError):
            respond(compiled, "  ", 5, np.random.default_rng(1))

    def test_length_is_in_tokens(self, compiled: Compiled) -> None:
        out = respond(compiled, "the", 4, np.random.default_rng(2))
        assert len(compiled.tokenizer.encode(out)) <= 8
        assert respond(compiled, "the", 0, np.random.default_rng(2)) == ""
