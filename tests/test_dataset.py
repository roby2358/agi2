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
            "for all curriculum stages to work properly."
        )
        self.temp_file.close()

    def teardown_method(self) -> None:
        os.unlink(self.temp_file.name)

    def test_initialization(self) -> None:
        dataset = TextDataset(self.temp_file.name, self.tokenizer, 10, 1)
        assert dataset.sources == [self.temp_file.name]
        assert dataset.tokenizer == self.tokenizer
        assert dataset.seq_len == 10

    def test_stage1_single_token_target(self) -> None:
        """Stage 1 should produce single-token targets."""
        dataset = TextDataset(self.temp_file.name, self.tokenizer, 20, 1)
        assert len(dataset) > 0
        item = dataset[0]
        assert "prompt_ids" in item
        assert "target_ids" in item
        assert item["target_ids"].shape[0] == 1

    def test_stage2_short_continuation(self) -> None:
        """Stage 2 should produce 2-5 token targets."""
        dataset = TextDataset(self.temp_file.name, self.tokenizer, 20, 2)
        assert len(dataset) > 0
        item = dataset[0]
        assert "prompt_ids" in item
        assert "target_ids" in item
        target_len = item["target_ids"].shape[0]
        assert 2 <= target_len <= 5

    def test_stage3_full_response(self) -> None:
        """Stage 3 should produce longer targets."""
        dataset = TextDataset(self.temp_file.name, self.tokenizer, 20, 3)
        assert len(dataset) > 0
        item = dataset[0]
        assert "prompt_ids" in item
        assert "target_ids" in item
        assert item["target_ids"].shape[0] > 1

    def test_set_stage(self) -> None:
        """set_stage should regenerate sequences."""
        dataset = TextDataset(self.temp_file.name, self.tokenizer, 20, 1)
        stage1_len = len(dataset)

        dataset.set_stage(3)
        assert dataset.stage == 3
        # Sequences are regenerated (may have different count)
        assert len(dataset) > 0

    def test_getitem_returns_dict(self) -> None:
        """__getitem__ should return a dict with tensor values."""
        dataset = TextDataset(self.temp_file.name, self.tokenizer, 10, 1)
        item = dataset[0]
        assert isinstance(item, dict)
        assert item["prompt_ids"].dtype == item["target_ids"].dtype

    def test_corpus_stats_includes_stage(self) -> None:
        """get_corpus_stats should include stage info."""
        dataset = TextDataset(self.temp_file.name, self.tokenizer, 10, 2)
        stats = dataset.get_corpus_stats()
        assert stats["stage"] == 2

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
                1,
            )
            assert len(dataset.sources) == 2
            assert len(dataset) > 0
        finally:
            os.unlink(temp2.name)
