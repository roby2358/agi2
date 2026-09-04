"""Tests for generation functions."""

import pytest
import torch
from src.basic_tokenizer import BasicTokenizer
from src.bpe_tokenizer import BPETokenizer
from src.generation import (
    SPECIAL_TOKEN_NAMES,
    STOP_TOKEN_NAMES,
    _apply_repetition_penalty,
    _apply_top_k,
    _ban_repeated_ngrams,
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
class TestRepetitionPenalty:
    def test_score_drops_proportionally_to_count(self):
        scores = torch.zeros(4)
        counts = torch.tensor([0.0, 1.0, 2.0, 3.0])
        out = _apply_repetition_penalty(scores, counts, 0.3, 0.3)
        # penalty * count / temperature: 0.3/0.3 = 1.0 per occurrence
        assert torch.allclose(out, torch.tensor([0.0, -1.0, -2.0, -3.0]))

    def test_zero_counts_leave_scores_unchanged(self):
        scores = torch.randn(8)
        out = _apply_repetition_penalty(scores, torch.zeros(8), 0.5, 0.3)
        assert torch.equal(out, scores)

    def test_ngram_ban_masks_completing_token(self):
        # Sequence contains trigram (1, 2, 3); with the last two tokens
        # being (1, 2) again, token 3 must be banned.
        scores = torch.zeros(5)
        out = _ban_repeated_ngrams(scores, [1, 2, 3, 4, 1, 2], 3)
        assert out[3] == -float("inf")
        assert torch.isfinite(out[[0, 1, 2, 4]]).all()

    def test_ngram_ban_skipped_when_it_would_ban_everything(self):
        # Tiny vocab where the ban would eliminate the only finite score:
        # better a repeat than no sample at all.
        scores = torch.full((3,), -float("inf"))
        scores[2] = 1.0
        out = _ban_repeated_ngrams(scores, [0, 1, 2, 0, 1], 3)
        assert torch.isfinite(out[2])

    def test_ngram_ban_off_by_default_size(self):
        scores = torch.zeros(5)
        assert torch.equal(_ban_repeated_ngrams(scores, [1, 2, 3, 1, 2], 0), scores)

    def test_generation_stops_stuttering_under_penalty(self):
        # A stub model whose hidden state always matches token 5's
        # embedding: unpenalized sampling at low temperature repeats 5
        # forever; the penalty must force diversity.
        class StubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.token_embeddings = torch.nn.modules.container.ModuleDict()
                emb = torch.nn.Embedding(10, 4)
                with torch.no_grad():
                    emb.weight.copy_(torch.eye(10, 4))
                    emb.weight[5] = torch.tensor([1.0, 1.0, 1.0, 1.0])
                holder = torch.nn.Module()
                holder.embedding = emb
                self.token_embeddings = holder

            def _run_transformer(self, ids):
                b, t = ids.shape
                return torch.ones(b, t, 4)

            def to(self, device):
                return self

            def eval(self):
                return self

        class StubTokenizer:
            vocab = {"<EOS>": 0}
            vocab_size = 10

            def encode(self, text):
                return [1]

            def decode(self, ids):
                return " ".join(str(i) for i in ids)

        torch.manual_seed(0)
        stuttered = generate_text(
            StubModel(), "x", 20, 0.05, 3, 1.0, StubTokenizer(), "cpu"
        )
        torch.manual_seed(0)
        penalized = generate_text(
            StubModel(),
            "x",
            20,
            0.05,
            3,
            1.0,
            StubTokenizer(),
            "cpu",
            repetition_penalty=2.0,
        )
        stuttered_counts = stuttered.split().count("5")
        penalized_counts = penalized.split().count("5")
        assert stuttered_counts > 15  # the stub really does stutter
        assert penalized_counts < stuttered_counts / 2


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
