"""Tests for training resume functionality."""

import pytest
import torch
import tempfile
import os
import shutil
import warnings
from pathlib import Path

from src.model import AGI2Model
from src.config import AGI2Config
from src.tokenizer import BasicTokenizer
from src.dataset import TextDataset
from src.training import train_model
from src.utils import save_checkpoint, load_checkpoint


class TestTrainingResume:
    """Test cases for training resume functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = AGI2Config(
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=64,
            n_positions=128,
            n_ctx=128
        )
        self.model = AGI2Model(self.config)
        
        self.tokenizer = BasicTokenizer()
        self.tokenizer.fit(["hello world test"])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("hello world test " * 100)  # Simple repeating text
            self.corpus_path = f.name
        
        self.checkpoint_path = None
        self.final_path = None
    
    def test_resume_training_basic(self):
        """Test basic resume training functionality"""
        # Suppress PyTorch security warnings for tests
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
            
            # Create a simple model
            config = AGI2Config(
                vocab_size=100,
                n_positions=64,
                n_embd=64,
                n_layer=2,
                n_head=2
            )
            model = AGI2Model(config)
            
            # Create a simple tokenizer
            tokenizer = BasicTokenizer()
            tokenizer.fit(["hello world test"])
            
            # Create a temporary corpus file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("hello world test " * 100)  # Simple repeating text
                corpus_path = f.name
            
            # Initialize variables to avoid UnboundLocalError
            checkpoint_path = None
            final_path = None
            
            try:
                # Create a temporary save path
                with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                    save_path = f.name
                
                # Train for 5 epochs to trigger checkpoint creation
                history1 = train_model(
                    model=model,
                    tokenizer=tokenizer,
                    corpus_path=corpus_path,
                    epochs=5,
                    batch_size=2,
                    seq_len=32,
                    device='cpu',
                    save_path=save_path
                )
                
                # Check that checkpoint was created
                # Extract model name from save_path for new directory structure
                model_name = os.path.basename(save_path).replace('.pt', '')
                checkpoint_path = f"trained/{model_name}.pt_epoch_5.pt"
                assert os.path.exists(checkpoint_path), "Checkpoint should be created"
                
                # Load checkpoint and verify contents
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                assert 'model_state_dict' in checkpoint, "Checkpoint should contain model state"
                assert 'epoch' in checkpoint, "Checkpoint should contain epoch"
                assert checkpoint['epoch'] == 5, "Checkpoint should have correct epoch"
                
                # Create new model instance
                model2 = AGI2Model(config)
                
                # Resume training for 1 more epoch
                history2 = train_model(
                    model=model2,
                    tokenizer=tokenizer,
                    corpus_path=corpus_path,
                    epochs=1,
                    batch_size=2,
                    seq_len=32,
                    device='cpu',
                    save_path=save_path,
                    start_epoch=5
                )
                
                # Check that training continued from epoch 6
                assert len(history2['train_loss']) == 1, "Should have trained for 1 epoch"
                
                # Check final model was saved
                final_path = f"trained/{model_name}.pt"
                assert os.path.exists(final_path), "Final model should be saved"
                
                final_checkpoint = torch.load(final_path, map_location='cpu', weights_only=False)
                assert final_checkpoint['epoch'] == 6, "Final model should have epoch 6"
                
            finally:
                # Cleanup
                if os.path.exists(corpus_path):
                    os.unlink(corpus_path)
                if os.path.exists(save_path):
                    os.unlink(save_path)
                if checkpoint_path and os.path.exists(checkpoint_path):
                    os.unlink(checkpoint_path)
                if final_path and os.path.exists(final_path):
                    os.unlink(final_path)
    
    def test_resume_training_invalid_checkpoint(self):
        """Test resume training with invalid checkpoint path"""
        config = AGI2Config(
            vocab_size=100,
            n_positions=64,
            n_embd=64,
            n_layer=2,
            n_head=2
        )
        model = AGI2Model(config)
        
        tokenizer = BasicTokenizer()
        tokenizer.fit(["hello world"])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("hello world " * 50)
            corpus_path = f.name
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                save_path = f.name
            
            # This should work (no resume)
            history = train_model(
                model=model,
                tokenizer=tokenizer,
                corpus_path=corpus_path,
                epochs=1,
                batch_size=2,
                seq_len=32,
                device='cpu',
                save_path=save_path,
                start_epoch=0
            )
            
            assert len(history['train_loss']) == 1, "Should train for 1 epoch"
            
        finally:
            if os.path.exists(corpus_path):
                os.unlink(corpus_path)
            if os.path.exists(save_path):
                os.unlink(save_path)


if __name__ == "__main__":
    pytest.main([__file__])
