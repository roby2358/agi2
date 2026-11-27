"""Tests for src/corpus.py."""

import pytest
from src.corpus import normalize, split_stream, tokenize, tokenize_split


@pytest.mark.unit
class TestNormalize:
    def test_markers_and_newlines_become_one_space(self) -> None:
        text = "# Title<|endoftext|>\n\nFirst  line.<|endoftext|>\nSecond.\n"
        assert normalize(text) == "# Title First line. Second."

    def test_no_marker_survives(self) -> None:
        assert "<|" not in normalize("a<|endoftext|>b")


STREAM = " ".join(f"word{i}" for i in range(400))


@pytest.mark.unit
class TestSplitStream:
    def test_every_word_lands_on_exactly_one_side(self) -> None:
        train, held_out = split_stream(STREAM, 0.25, block_chars=100, seed=1)
        assert sorted((train + held_out).split()) == sorted(STREAM.split())
        assert 0.15 < len(held_out) / len(STREAM) < 0.35

    def test_blocks_stay_intact_and_in_order(self) -> None:
        train, held_out = split_stream(STREAM, 0.25, block_chars=100, seed=1)
        for part in (train, held_out):
            numbers = [int(word[4:]) for word in part.split()]
            assert numbers == sorted(numbers)
            assert "  " not in part
        # Held-out blocks come from across the stream, not one end.
        held = [int(word[4:]) for word in held_out.split()]
        assert held[0] < 200 < held[-1]

    def test_seed_chooses_the_blocks(self) -> None:
        first = split_stream(STREAM, 0.25, block_chars=100, seed=1)
        assert first == split_stream(STREAM, 0.25, block_chars=100, seed=1)
        assert first != split_stream(STREAM, 0.25, block_chars=100, seed=2)

    def test_zero_fraction_holds_nothing_out(self) -> None:
        assert split_stream("a b c", 0.0) == ("a b c", "")

    def test_a_single_block_holds_nothing_out(self) -> None:
        assert split_stream("abcdefgh ij", 0.5, block_chars=1000) == ("abcdefgh ij", "")

    def test_training_text_is_never_empty(self) -> None:
        train, held_out = split_stream(STREAM, 0.99, block_chars=100, seed=0)
        assert train and held_out


@pytest.mark.integration
class TestTokenize:
    def test_round_trip_and_range(self) -> None:
        text = "the cat sat on the mat.\nthe dog sat on the log.<|endoftext|>\n" * 20
        tokens, tokenizer = tokenize(text, 300)
        assert tokenizer.decode(tokens) == normalize(text)
        assert 0 <= min(tokens) and max(tokens) < tokenizer.vocab_size

    def test_split_keeps_held_out_text_from_the_tokenizer(self) -> None:
        text = "the cat sat on the mat. " * 40
        train, held_out, tokenizer = tokenize_split(
            text, 300, 0.2, holdout_block=100, holdout_seed=0
        )
        assert train and held_out
        whole, _ = tokenize(text, 300)
        assert len(train) < len(whole)
        decoded = tokenizer.decode(train) + tokenizer.decode(held_out)
        assert sorted(decoded.split()) == sorted(normalize(text).split())
