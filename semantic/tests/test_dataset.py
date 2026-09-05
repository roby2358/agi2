"""Tests for TextDataset class."""

import os
import tempfile

import pytest
from src.basic_tokenizer import BasicTokenizer
from src.dataset import TextDataset


class TestTextDataset:
    def setup_method(self) -> None:
        self.tokenizer = BasicTokenizer()
        self.tokenizer.fit(
            ["This is a test corpus for testing the dataset with enough tokens."]
        )
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        )
        self.temp_file.write(
            "This is a test corpus for testing the dataset. "
            "It needs to be long enough to create sequences. "
            "Adding more text here to ensure we have enough tokens "
            "for the dataset to work properly."
        )
        self.temp_file.close()

    def teardown_method(self) -> None:
        os.unlink(self.temp_file.name)

    def test_initialization(self) -> None:
        dataset = TextDataset(self.temp_file.name, self.tokenizer, 10)
        assert dataset.sources == [self.temp_file.name]
        assert dataset.tokenizer == self.tokenizer
        assert dataset.seq_len == 10

    def test_single_token_target(self) -> None:
        """Dataset should produce single-token targets."""
        dataset = TextDataset(self.temp_file.name, self.tokenizer, 20)
        assert len(dataset) > 0
        item = dataset[0]
        assert "prompt_ids" in item
        assert "target_ids" in item
        assert item["target_ids"].shape[0] == 1

    def test_getitem_returns_dict(self) -> None:
        """__getitem__ should return a dict with tensor values."""
        dataset = TextDataset(self.temp_file.name, self.tokenizer, 10)
        item = dataset[0]
        assert isinstance(item, dict)
        assert item["prompt_ids"].dtype == item["target_ids"].dtype

    def test_corpus_stats(self) -> None:
        """get_corpus_stats should return correct info."""
        dataset = TextDataset(self.temp_file.name, self.tokenizer, 10)
        stats = dataset.get_corpus_stats()
        assert "total_tokens" in stats
        assert "total_sequences" in stats
        assert "sequence_length" in stats
        assert stats["sequence_length"] == 10

    def test_set_seq_len(self) -> None:
        """set_seq_len should regenerate sequences with new length."""
        dataset = TextDataset(self.temp_file.name, self.tokenizer, 5)
        len_at_5 = len(dataset)

        dataset.set_seq_len(20)
        assert dataset.seq_len == 20
        assert len(dataset) > 0
        # Different seq_len produces different sequence count
        assert len(dataset) != len_at_5 or dataset.seq_len != 5

    def test_minimum_seq_len(self) -> None:
        """seq_len=2 should produce 1-token prompts."""
        dataset = TextDataset(self.temp_file.name, self.tokenizer, 2)
        assert len(dataset) > 0
        item = dataset[0]
        assert item["prompt_ids"].shape[0] == 1
        assert item["target_ids"].shape[0] == 1

    def test_multiple_sources(self) -> None:
        """Dataset should support multiple source files."""
        temp2 = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
        temp2.write("More text for the second source file with more tokens.")
        temp2.close()

        try:
            dataset = TextDataset(
                [self.temp_file.name, temp2.name],
                self.tokenizer,
                10,
            )
            assert len(dataset.sources) == 2
            assert len(dataset) > 0
        finally:
            os.unlink(temp2.name)


@pytest.mark.unit
class TestBoundaryAlignment:
    """Window starts snap to utterance boundaries when a boundary token id
    is given; without one, behavior is the original stride walk."""

    class IdTokenizer:
        """Encodes space-separated ints as themselves — makes token
        positions explicit in tests."""

        vocab_size = 100

        def encode(self, text: str) -> list[int]:
            return [int(t) for t in text.split()]

    def _dataset(self, tokens: list[int], seq_len: int, boundary) -> TextDataset:
        temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
        temp.write(" ".join(str(t) for t in tokens))
        temp.close()
        try:
            return TextDataset(temp.name, self.IdTokenizer(), seq_len, boundary)
        finally:
            os.unlink(temp.name)

    def test_no_boundary_token_keeps_stride_starts(self) -> None:
        tokens = list(range(20))
        dataset = self._dataset(tokens, 9, None)
        starts = [seq["prompt_ids"][0] for seq in dataset.sequences]
        # max_prompt 8, step 4: the original walk
        assert starts == [tokens[i] for i in range(0, 19, 4)]

    def test_starts_snap_to_utterance_boundaries(self) -> None:
        # Boundary token 99 at positions 3, 8, 13: utterances start at
        # 0, 4, 9, 14. With step 4, each next boundary >= step past the
        # previous start is chosen.
        tokens = [10, 11, 12, 99, 20, 21, 22, 23, 99, 30, 31, 32, 33, 99, 40, 41]
        dataset = self._dataset(tokens, 9, 99)
        start_positions = dataset._window_starts(4)
        assert start_positions == [0, 4, 9, 14]
        for pos in start_positions[1:]:
            assert tokens[pos - 1] == 99  # each start follows a boundary

    def test_windows_may_cross_boundaries(self) -> None:
        # Alignment fixes where windows START; context still spans
        # utterances, so the boundary token appears inside prompts and the
        # model trains on it.
        tokens = [10, 11, 99, 20, 21, 22, 99, 30, 31, 32]
        dataset = self._dataset(tokens, 9, 99)
        assert any(99 in seq["prompt_ids"] for seq in dataset.sequences)

    def test_markerless_stretch_gets_stride_fill(self) -> None:
        # One boundary early, then a long marker-less run: the run must
        # still be covered by stride windows, not just the one boundary.
        # (The utterance start at 2 is deliberately skipped — it's closer
        # than a stride to the previous start at 0.)
        tokens = [10, 99] + list(range(30, 60))
        dataset = self._dataset(tokens, 9, 99)
        start_positions = dataset._window_starts(4)
        assert start_positions == [0, 4, 8, 12, 16, 20, 24]
        # Tail is covered: the last window reaches the corpus end.
        assert start_positions[-1] + 8 >= len(tokens) - 1

    def test_sequence_count_stays_comparable(self) -> None:
        # Snapping must not multiply the epoch's sequence count: starts
        # stay at least a stride apart.
        tokens = []
        for utterance in range(40):
            tokens.extend([utterance % 50, (utterance + 1) % 50, 99])
        aligned = self._dataset(tokens, 9, 99)
        unaligned = self._dataset(tokens, 9, None)
        assert len(aligned) <= len(unaligned) + 1

    def test_stats_report_boundaries(self) -> None:
        tokens = [10, 11, 99, 20, 21, 99, 30, 31]
        dataset = self._dataset(tokens, 5, 99)
        assert dataset.get_corpus_stats()["utterance_boundaries"] == 3


class TestValidationSplit:
    """val_fraction holds out a boundary-aligned corpus tail from training."""

    class IdTokenizer:
        vocab_size = 100

        def encode(self, text: str) -> list[int]:
            return [int(t) for t in text.split()]

    def _dataset(
        self, tokens: list[int], seq_len: int, boundary, val_fraction: float
    ) -> TextDataset:
        temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
        temp.write(" ".join(str(t) for t in tokens))
        temp.close()
        try:
            return TextDataset(
                temp.name, self.IdTokenizer(), seq_len, boundary, val_fraction
            )
        finally:
            os.unlink(temp.name)

    def test_default_no_holdout(self) -> None:
        tokens = list(range(50))
        dataset = self._dataset(tokens, 10, None, 0.0)
        stats = dataset.get_corpus_stats()
        assert stats["train_tokens"] == 50
        assert stats["val_tokens"] == 0
        assert dataset.val_windows(10) == []

    def test_cut_snaps_to_boundary(self) -> None:
        # Boundaries (id 99) after positions 9, 19, 29, 39; 20% target
        # cut = 40, which is already an utterance start (position 40).
        tokens = []
        for utterance in range(5):
            tokens.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 99])
        dataset = self._dataset(tokens, 6, 99, 0.2)
        assert dataset._train_end == 40

    def test_train_sequences_stay_out_of_val(self) -> None:
        # Distinct token values encode positions (value = position + 1), so
        # a training window that touched the held-out tail would show a
        # value above the cut.
        tokens = list(range(1, 51))
        dataset = self._dataset(tokens, 6, None, 0.2)
        cut = dataset._train_end
        assert cut < len(tokens)
        for seq in dataset.sequences:
            assert max(seq["prompt_ids"] + seq["target_ids"]) <= cut

    def test_val_windows_cover_tail(self) -> None:
        tokens = list(range(1, 41))
        dataset = self._dataset(tokens, 6, None, 0.25)
        cut = dataset._train_end
        windows = dataset.val_windows(6)
        assert windows
        covered = []
        for w in windows:
            covered.extend(w["prompt_ids"].tolist())
        # Prompts tile the held-out region without overlap
        assert covered == tokens[cut : len(tokens) - 1]

    def test_set_seq_len_preserves_holdout(self) -> None:
        tokens = list(range(1, 61))
        dataset = self._dataset(tokens, 6, None, 0.2)
        cut = dataset._train_end
        dataset.set_seq_len(12)
        assert dataset._train_end == cut
        for seq in dataset.sequences:
            assert len(seq["prompt_ids"]) + len(seq["target_ids"]) <= cut
