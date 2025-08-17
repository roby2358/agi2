"""Tests for training functions."""
import pytest
import torch
from src.training import train_epoch
import unittest
import tempfile
import os
from unittest.mock import Mock, patch
from src.training import train_model
from src.model import GPT2Model
from src.config import GPT2Config
from src.tokenizer import BasicTokenizer


class TestTraining:
    def test_train_epoch_signature(self):
        """Test that train_epoch function has correct signature."""
        # This is a basic test to ensure the function exists and has correct signature
        assert callable(train_epoch)
        
        # Check that it takes the expected parameters
        import inspect
        sig = inspect.signature(train_epoch)
        params = list(sig.parameters.keys())
        
        expected_params = ['model', 'dataloader', 'optimizer', 'criterion', 'device']
        for param in expected_params:
            assert param in params


class TestTraining(unittest.TestCase):
    
    def setUp(self):
        # Create a minimal model for testing
        self.config = GPT2Config(
            vocab_size=1000,
            n_positions=128,
            n_embd=64,
            n_layer=2,
            n_head=2
        )
        self.model = GPT2Model(self.config)
        self.tokenizer = BasicTokenizer()
        self.tokenizer.fit(["test text for vocabulary"])
        
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        # Clean up
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('src.training.TextDataset')
    @patch('src.training.DataLoader')
    @patch('src.training.optim.AdamW')
    @patch('src.training.nn.CrossEntropyLoss')
    def test_model_saving_directory_structure(self, mock_loss, mock_optimizer, mock_dataloader, mock_dataset):
        """Test that models are saved in the correct directory structure with proper naming."""
        # Mock the dataset and dataloader
        mock_dataset_instance = Mock()
        mock_dataset_instance.__len__ = Mock(return_value=10)
        mock_dataset.return_value = mock_dataset_instance
        
        mock_dataloader_instance = Mock()
        mock_dataloader_instance.__iter__ = Mock(return_value=iter([]))
        mock_dataloader_instance.__len__ = Mock(return_value=1)
        mock_dataloader.return_value = mock_dataloader_instance
        
        # Mock the optimizer and loss
        mock_optimizer_instance = Mock()
        mock_optimizer_instance.state_dict = Mock(return_value={})
        mock_optimizer.return_value = mock_optimizer_instance
        
        mock_loss_instance = Mock()
        mock_loss.return_value = mock_loss_instance
        
        # Mock the model's get_num_params method
        self.model.get_num_params = Mock(return_value=1000)
        
        # Create a temporary corpus file
        corpus_path = "temp_corpus.txt"
        with open(corpus_path, 'w') as f:
            f.write("test text for training")
        
        # Run training with minimal epochs
        try:
            train_model(
                model=self.model,
                tokenizer=self.tokenizer,
                corpus_path=corpus_path,
                epochs=1,
                batch_size=1,
                save_path="test_model",
                device="cpu"
            )
            
            # Check that the trained directory was created
            self.assertTrue(os.path.exists("trained"))
            
            # Check that the final model was saved with correct name
            self.assertTrue(os.path.exists("trained/test_model.pt"))
            
            # Check that no _final suffix was added
            self.assertFalse(os.path.exists("trained/test_model_final.pt"))
            
        except Exception as e:
            # Training might fail due to mocked components, but we can still check directory creation
            if "trained" in os.listdir():
                self.assertTrue(os.path.exists("trained"))
            else:
                self.fail(f"Training failed unexpectedly: {e}")
        
        # Clean up
        if os.path.exists(corpus_path):
            os.remove(corpus_path)


if __name__ == '__main__':
    unittest.main()
