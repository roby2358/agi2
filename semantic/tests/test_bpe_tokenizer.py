"""Tests for BPETokenizer class."""

import pickle

import pytest
from src.bpe_tokenizer import BPETokenizer

CORPUS = [
    "What ho! Give ear to him whose words our speech did frame.",
    "In joy and grief alike, we speak his name.",
    "Thou art more lovely and more temperate.",
]


@pytest.mark.unit
class TestBPETokenizer:
    def test_initialization(self):
        """Test BPETokenizer initialization."""
        tokenizer = BPETokenizer(vocab_size=1000)
        assert tokenizer.vocab_size == 1000
        assert tokenizer.vocab == {}

    def test_initialization_default_vocab_size(self):
        """Test BPETokenizer initialization with default vocab_size."""
        tokenizer = BPETokenizer()
        assert tokenizer.vocab_size == 4096

    def test_unfitted_raises(self):
        """Encoding before fit must fail loudly, not silently."""
        tokenizer = BPETokenizer()
        with pytest.raises(RuntimeError, match="must be fitted"):
            tokenizer.encode("hello")
        with pytest.raises(RuntimeError, match="must be fitted"):
            tokenizer.decode([0])

    def test_fit_builds_vocabulary(self):
        """fit trains a vocabulary of roughly the requested size."""
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.fit(CORPUS)

        # Byte-level alphabet + specials is the floor; target is the ceiling
        assert 257 <= tokenizer.vocab_size <= 300
        for token in BPETokenizer.SPECIAL_TOKENS:
            assert token in tokenizer.vocab
            assert tokenizer.vocab[token] >= 0
        assert len(set(tokenizer.vocab.values())) == len(BPETokenizer.SPECIAL_TOKENS)

    def test_special_tokens_encode_atomically(self):
        """Registered specials encode to a single id even inside text —
        the corpus's literal <|endoftext|> markers become one real token."""
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.fit(["speak the speech<|endoftext|>" * 20])

        for token in BPETokenizer.SPECIAL_TOKENS:
            assert tokenizer.encode(token) == [tokenizer.vocab[token]]

        ids = tokenizer.encode("the speech<|endoftext|>")
        assert ids.count(tokenizer.vocab["<|endoftext|>"]) == 1
        assert tokenizer.decode(ids) == "the speech<|endoftext|>"

    def test_encode_decode_roundtrip(self):
        """Byte-level BPE must round-trip corpus text exactly."""
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.fit(CORPUS)

        for text in CORPUS:
            assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_roundtrip_unseen_characters(self):
        """Text with characters never seen in training still round-trips
        (byte-level alphabet covers everything)."""
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.fit(CORPUS)

        text = "Zounds! 42 éüñ 世界"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_merges_learned(self):
        """A trained vocab must contain multi-byte merged tokens, not just
        the byte alphabet."""
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.fit(CORPUS)
        # 256 bytes + <EOS> = 257; anything above that is a learned merge
        assert tokenizer.vocab_size > 257

    def test_compression(self):
        """Trained merges should compress corpus text below one token per
        character."""
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.fit(CORPUS)
        text = CORPUS[0]
        assert len(tokenizer.encode(text)) < len(text)

    def test_pickle_roundtrip(self):
        """The tokenizer rides inside torch checkpoints via pickle."""
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.fit(CORPUS)

        restored = pickle.loads(pickle.dumps(tokenizer))

        text = CORPUS[1]
        assert restored.encode(text) == tokenizer.encode(text)
        assert restored.decode(restored.encode(text)) == text
        assert restored.vocab_size == tokenizer.vocab_size
        assert restored.vocab == tokenizer.vocab

    def test_save_load_vocab(self, tmp_path):
        """save_vocab/load_vocab persist the trained tokenizer to JSON."""
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.fit(CORPUS)

        path = str(tmp_path / "bpe.json")
        tokenizer.save_vocab(path)

        loaded = BPETokenizer()
        loaded.load_vocab(path)

        text = CORPUS[2]
        assert loaded.encode(text) == tokenizer.encode(text)
        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded.vocab == tokenizer.vocab
