"""Tests for generation functions."""

import pytest
import torch
from src.basic_tokenizer import BasicTokenizer
from src.bpe_tokenizer import BPETokenizer
from src.generation import (
    SPECIAL_TOKEN_NAMES,
    STOP_TOKEN_NAMES,
    _apply_top_k,
    _decode_stripped,
    _token_ids,
    build_corpus_token_mask,
    generate_text,
)


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
class TestStopAndSpecialTokens:
    def _bpe(self) -> BPETokenizer:
        tokenizer = BPETokenizer(vocab_size=400)
        tokenizer.fit(["to be or not to be.<|endoftext|>" * 30])
        return tokenizer

    def test_stop_ids_cover_eos_and_endoftext(self):
        tokenizer = self._bpe()
        stop = _token_ids(tokenizer, STOP_TOKEN_NAMES)
        assert tokenizer.vocab["<EOS>"] in stop
        assert tokenizer.vocab["<|endoftext|>"] in stop
        assert tokenizer.vocab["<|pad|>"] not in stop
        assert tokenizer.vocab["<|break|>"] not in stop

    def test_basic_tokenizer_only_defines_eos(self):
        # BasicTokenizer has <EOS> but no <|endoftext|>; the lookup must
        # simply skip names a tokenizer doesn't define.
        tokenizer = BasicTokenizer()
        tokenizer.fit(["abc"])
        stop = _token_ids(tokenizer, STOP_TOKEN_NAMES)
        assert stop == {tokenizer.vocab["<EOS>"]}

    def test_decode_strips_special_tokens(self):
        # Special tokens belong in training, never in inference output.
        tokenizer = self._bpe()
        ids = tokenizer.encode("to be<|endoftext|>")
        assert tokenizer.vocab["<|endoftext|>"] in ids
        text = _decode_stripped(tokenizer, ids)
        assert "<|endoftext|>" not in text
        assert "to be" in text

    def test_mask_allows_endoftext(self, tmp_path):
        tokenizer = self._bpe()
        source = tmp_path / "corpus.txt"
        source.write_text("to be or not", encoding="utf-8")
        mask = build_corpus_token_mask(
            [str(source)], tokenizer, tokenizer.vocab_size, "cpu"
        )
        assert mask is not None
        for name in STOP_TOKEN_NAMES:
            assert mask[tokenizer.vocab[name]]

    def test_special_names_are_superset_of_stop_names(self):
        assert set(STOP_TOKEN_NAMES) <= set(SPECIAL_TOKEN_NAMES)


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
