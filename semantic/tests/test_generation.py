"""Tests for generation functions."""

import pytest
import torch
from src.basic_tokenizer import BasicTokenizer
from src.generation import _apply_top_k, build_corpus_token_mask, generate_text


class TestGeneration:
    def test_generate_text_signature(self):
        """Test that generate_text function has correct signature."""
        assert callable(generate_text)

        # Check that it takes the expected parameters
        import inspect

        sig = inspect.signature(generate_text)
        params = list(sig.parameters.keys())

        expected_params = [
            "model",
            "prompt",
            "max_length",
            "temperature",
            "top_k",
            "top_p",
            "tokenizer",
            "device",
        ]
        for param in expected_params:
            assert param in params


@pytest.mark.unit
class TestCorpusTokenMask:
    def _make_tokenizer(self, text: str) -> BasicTokenizer:
        tokenizer = BasicTokenizer()
        tokenizer.fit([text])
        return tokenizer

    def test_mask_marks_only_corpus_tokens(self, tmp_path):
        """Tokens from the corpus (plus <EOS>) are allowed; others are not."""
        corpus = "abc"
        tokenizer = self._make_tokenizer(corpus + "xyz")

        source = tmp_path / "corpus.txt"
        source.write_text(corpus, encoding="utf-8")

        mask = build_corpus_token_mask(
            [str(source)], tokenizer, tokenizer.vocab_size, "cpu"
        )

        assert mask is not None
        assert mask.dtype == torch.bool
        assert mask.shape == (tokenizer.vocab_size,)

        for char in corpus:
            assert mask[tokenizer.vocab[char]]
        assert mask[tokenizer.vocab["<EOS>"]]
        for char in "xyz":
            assert not mask[tokenizer.vocab[char]]

    def test_missing_sources_return_none(self, tmp_path):
        """All sources missing means no mask (unrestricted generation)."""
        tokenizer = self._make_tokenizer("abc")
        mask = build_corpus_token_mask(
            [str(tmp_path / "nope.txt")], tokenizer, tokenizer.vocab_size, "cpu"
        )
        assert mask is None

    def test_missing_source_skipped(self, tmp_path):
        """A missing source is skipped; existing sources still build the mask."""
        tokenizer = self._make_tokenizer("abc")
        source = tmp_path / "corpus.txt"
        source.write_text("a", encoding="utf-8")

        mask = build_corpus_token_mask(
            [str(tmp_path / "nope.txt"), str(source)],
            tokenizer,
            tokenizer.vocab_size,
            "cpu",
        )

        assert mask is not None
        assert mask[tokenizer.vocab["a"]]
        assert not mask[tokenizer.vocab["b"]]


@pytest.mark.unit
class TestApplyTopK:
    def test_top_k_larger_than_vocab_is_clamped(self):
        """top_k larger than the vocab must not crash."""
        scores = torch.tensor([0.1, 0.5, 0.3])
        filtered = _apply_top_k(scores, 50)
        assert torch.equal(filtered, scores)

    def test_top_k_filters_low_scores(self):
        scores = torch.tensor([0.1, 0.5, 0.3])
        filtered = _apply_top_k(scores, 1)
        assert filtered[1] == pytest.approx(0.5)
        assert filtered[0] == -float("inf")
        assert filtered[2] == -float("inf")
