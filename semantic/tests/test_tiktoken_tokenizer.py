"""Tests for TiktokenTokenizer wrapper."""

import pytest

from src.tiktoken_tokenizer import TiktokenTokenizer


class TestTiktokenTokenizer:
    def setup_method(self):
        self.tok = TiktokenTokenizer()

    def test_vocab_size(self):
        assert self.tok.vocab_size == 50257

    def test_encode_decode_roundtrip(self):
        text = "Hello, world!"
        ids = self.tok.encode(text)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert self.tok.decode(ids) == text

    def test_encode_produces_subwords(self):
        text = "Hello, world!"
        ids = self.tok.encode(text)
        # BPE should produce fewer tokens than characters
        assert len(ids) < len(text)

    def test_eos_in_vocab(self):
        assert "<EOS>" in self.tok.vocab
        eos_id = self.tok.vocab["<EOS>"]
        assert isinstance(eos_id, int)
        assert 0 <= eos_id < self.tok.vocab_size

    def test_fit_is_noop(self):
        # fit should not raise or change state
        before = self.tok.vocab_size
        self.tok.fit(["some text"])
        assert self.tok.vocab_size == before

    def test_empty_string(self):
        ids = self.tok.encode("")
        assert ids == []
        assert self.tok.decode([]) == ""

    def test_unicode(self):
        text = "café résumé naïve"
        ids = self.tok.encode(text)
        assert self.tok.decode(ids) == text
