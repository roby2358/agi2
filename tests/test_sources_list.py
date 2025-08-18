"""
Test sources list functionality

This module tests the new sources list feature for training data.
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.config_loader import get_sources_list
from src.dataset import TextDataset
from src.tokenizer import BasicTokenizer


class TestSourcesList:
    """Test the sources list functionality."""
    
    def test_get_sources_list_new_format(self):
        """Test getting sources list from new 'sources' config."""
        config = {
            'sources': [
                'file1.txt',
                'file2.txt',
                'file3.txt'
            ]
        }
        
        sources = get_sources_list(config)
        assert sources == ['file1.txt', 'file2.txt', 'file3.txt']
    
    def test_get_sources_list_legacy_format(self):
        """Test getting sources list from legacy 'corpus_path' config."""
        config = {
            'corpus_path': 'legacy_corpus.txt'
        }
        
        sources = get_sources_list(config)
        assert sources == ['legacy_corpus.txt']
    
    def test_get_sources_list_empty_sources(self):
        """Test that empty sources list raises error."""
        config = {
            'sources': []
        }
        
        with pytest.raises(ValueError, match="Sources list cannot be empty"):
            get_sources_list(config)
    
    def test_get_sources_list_missing_both(self):
        """Test that missing both sources and corpus_path raises error."""
        config = {}
        
        with pytest.raises(ValueError, match="Configuration must contain either 'sources' list or 'corpus_path'"):
            get_sources_list(config)
    
    def test_dataset_with_sources_list(self):
        """Test TextDataset with multiple sources."""
        # Create temporary files with test content
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            file1_path = os.path.join(temp_dir, 'file1.txt')
            file2_path = os.path.join(temp_dir, 'file2.txt')
            
            with open(file1_path, 'w', encoding='utf-8') as f:
                f.write("Hello world from file 1.")
            
            with open(file2_path, 'w', encoding='utf-8') as f:
                f.write("Hello world from file 2.")
            
            # Create tokenizer
            tokenizer = BasicTokenizer()
            tokenizer.fit(["Hello world from file 1. Hello world from file 2."])
            
            # Test with sources list
            sources = [file1_path, file2_path]
            dataset = TextDataset(sources, tokenizer, seq_len=10)
            
            assert len(dataset.sources) == 2
            assert dataset.sources == sources
            assert len(dataset) > 0  # Should have some sequences
            
            # Test stats
            stats = dataset.get_corpus_stats()
            assert stats['num_sources'] == 2
            assert stats['sources'] == sources
    
    def test_dataset_with_single_source(self):
        """Test TextDataset with single source (backward compatibility)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'single_file.txt')
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("Hello world from single file.")
            
            tokenizer = BasicTokenizer()
            tokenizer.fit(["Hello world from single file."])
            
            # Test with single source
            dataset = TextDataset(file_path, tokenizer, seq_len=10)
            
            assert len(dataset.sources) == 1
            assert dataset.sources == [file_path]
            assert len(dataset) > 0
            
            # Test stats
            stats = dataset.get_corpus_stats()
            assert stats['num_sources'] == 1
            assert stats['sources'] == [file_path]


if __name__ == "__main__":
    pytest.main([__file__])
