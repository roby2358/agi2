"""Tests for src/pipeline.py."""

from pathlib import Path

import pytest
from src.config import HegelConfig
from src.pipeline import compile_from_config

TEXT = "the cat sat on the mat.\nthe dog sat on the log.<|endoftext|>\n" * 30


@pytest.mark.integration
class TestCompileFromConfig:
    def test_produces_a_consistent_run(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(TEXT)
        config = HegelConfig(corpus=str(corpus), vocab_size=280, max_context=2)
        compiled = compile_from_config(config)
        assert compiled.text_length == len(TEXT)
        assert compiled.chain.vocab_size == compiled.tokenizer.vocab_size
        assert compiled.chain.max_context == 2
        assert compiled.model.d_ff == len(compiled.chain.contexts) - 1
        assert compiled.model.n_heads == 1
