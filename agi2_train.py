#!/usr/bin/env python3
"""
AGI2 Training Script

Usage:
    python agi2_train.py <config_file>

Example:
    python agi2_train.py resources/moby_dick.toml
    python agi2_train.py resources/default.toml
"""

import sys
import torch
from pathlib import Path

from src.model import GPT2Model
from src.tokenizer import BasicTokenizer
from src.training import train_model
from src.config import GPT2Config
from src.utils import load_checkpoint
from src.cuda_utils import check_cuda_availability, get_optimal_device
from src.config_loader import get_training_config, get_config_value


def main():
    if len(sys.argv) != 2:
        print("Usage: python agi2_train.py <config_file>")
        print("Example: python agi2_train.py resources/moby_dick.toml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    try:
        # Load configuration from TOML file
        config = get_training_config(config_path)
        print(f"Loaded configuration from: {config_path}")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Check CUDA availability at startup
    print("Checking CUDA availability for training...")
    cuda_status = check_cuda_availability(verbose=True)
    
    # Print GPU memory monitoring command
    print("\nTo monitor GPU memory during training, use:")
    print("  nvidia-smi -l 1  # Updates every 1 second")
    print("  watch -n 1 nvidia-smi  # Updates every 1 second (Linux/Mac)")
    print("  # On Windows, use Task Manager > Performance > GPU\n")
    
    # Extract configuration values with defaults
    corpus_path = Path(get_config_value(config, 'corpus_path'))
    epochs = get_config_value(config, 'epochs', 10)
    batch_size = get_config_value(config, 'batch_size', 12)
    learning_rate = get_config_value(config, 'learning_rate', 3e-4)
    seq_len = get_config_value(config, 'seq_len', 1024)
    model_name = get_config_value(config, 'model_name', 'model')
    resume_path = get_config_value(config, 'resume')
    device_choice = get_config_value(config, 'device', 'auto')
    
    # Model architecture parameters (optional)
    model_positions = get_config_value(config, 'model_positions', 1024)
    model_embd = get_config_value(config, 'model_embd', 768)
    model_layer = get_config_value(config, 'model_layer', 12)
    model_head = get_config_value(config, 'model_head', 12)
    
    # Validate corpus path
    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {corpus_path}")
        sys.exit(1)
    
    # Validate resume checkpoint if provided
    if resume_path:
        resume_path = Path(resume_path)
        if not resume_path.exists():
            print(f"Error: Resume checkpoint not found: {resume_path}")
            sys.exit(1)
        print(f"Resuming training from checkpoint: {resume_path}")
    
    # Determine device using our utility
    device = get_optimal_device(device_choice)
    
    print(f"Using device: {device}")
    print(f"Training on corpus: {corpus_path}")
    print(f"Model name: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Sequence length: {seq_len}")
    
    # Initialize tokenizer and build vocabulary first
    tokenizer = BasicTokenizer()
    
    # Load a sample of text to build vocabulary
    print("Building vocabulary from corpus...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        sample_text = f.read()[:50000]  # First 50k characters for vocab building
    
    tokenizer.fit([sample_text])
    actual_vocab_size = tokenizer.vocab_size
    
    print(f"Vocabulary built with {actual_vocab_size} tokens")
    
    # Initialize model with correct vocabulary size
    model_config = GPT2Config(
        vocab_size=actual_vocab_size,  # Use actual vocab size from tokenizer
        n_positions=model_positions,
        n_embd=model_embd,
        n_layer=model_layer,
        n_head=model_head
    )
    
    model = GPT2Model(model_config)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_path:
        print(f"Loading checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model state from epoch {checkpoint['epoch']}")
        
        # Load tokenizer if it exists in checkpoint
        if 'tokenizer' in checkpoint:
            tokenizer = checkpoint['tokenizer']
            print("Loaded tokenizer from checkpoint")
        
        # Calculate starting epoch
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch + 1}")
    
    # Train the model
    try:
        training_history = train_model(
            model=model,
            tokenizer=tokenizer,  # Pass the fitted tokenizer
            corpus_path=str(corpus_path),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seq_len=seq_len,
            device=device,
            save_path=model_name,
            start_epoch=start_epoch  # Pass starting epoch
        )
        
        print(f"Training completed successfully!")
        print(f"Model saved to: trained/{model_name}.pt")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
