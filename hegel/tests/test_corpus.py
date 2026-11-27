"""Tests for src/corpus.py."""

import pytest
from src.corpus import normalize, tokenize


@pytest.mark.unit
class TestNormalize:
    def test_markers_and_newlines_become_one_space(self) -> None:
        text = "# Title<|endoftext|>\n\nFirst  line.<|endoftext|>\nSecond.\n"
        assert normalize(text) == "# Title First line. Second."

    def test_no_marker_survives(self) -> None:
        assert "<|" not in normalize("a<|endoftext|>b")


@pytest.mark.integration
class TestTokenize:
    def test_round_trip_and_range(self) -> None:
        text = "the cat sat on the mat.\nthe dog sat on the log.<|endoftext|>\n" * 20
        tokens, tokenizer = tokenize(text, 300)
        assert tokenizer.decode(tokens) == normalize(text)
        assert 0 <= min(tokens) and max(tokens) < tokenizer.vocab_size
