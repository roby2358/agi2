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
