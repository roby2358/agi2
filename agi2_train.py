#!/usr/bin/env python3
"""
AGI2 Training Script

This script trains an AGI2 model on text data.
Usage: python agi2_train.py <config_file>
"""

import sys
import torch
from pathlib import Path

import argparse
import logging
import os

from src.config_loader import get_training_config, get_config_value, get_sources_list
from src.cuda_utils import check_cuda_availability, get_optimal_device
from src.config import AGI2Config
from src.model import AGI2Model
from src.basic_tokenizer import BasicTokenizer
from src.training import train_model

def main():
    if len(sys.argv) < 2:
        print("Usage: python agi2_train.py <config_file>")
        print("Example: python agi2_train.py resources/small_model.toml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    try:
        # Load configuration from TOML file
        config = get_training_config(config_path)
        print(f"Loaded configuration from: {config_path}")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Check CUDA availability
    print("Checking CUDA availability for training...")
    cuda_status = check_cuda_availability(verbose=True)
    
    # Extract configuration values
    sources = get_sources_list(config)
    model_name = get_config_value(config, 'model_name')
    epochs = get_config_value(config, 'epochs', 10)
    batch_size = get_config_value(config, 'batch_size', 4)
    learning_rate = get_config_value(config, 'learning_rate', 1e-4)
    seq_len = get_config_value(config, 'seq_len', 512)
    device_choice = get_config_value(config, 'device', 'auto')
    resume_path = get_config_value(config, 'resume', '')
    
    # Validate source paths
    for source_path in sources:
        if not Path(source_path).exists():
            print(f"Error: Source file not found: {source_path}")
            sys.exit(1)
    
    print(f"Training with {len(sources)} source(s):")
    for source_path in sources:
        print(f"  - {source_path}")
    
    # Determine device
    device = get_optimal_device(device_choice)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs("trained", exist_ok=True)
    
    # Load a sample of text to build vocabulary
    print("Building vocabulary from corpus...")
    with open(sources[0], 'r', encoding='utf-8') as f: # Assuming the first source is the corpus for vocab building
        sample_text = f.read()[:50000]  # First 50k characters for vocab building
    
    tokenizer = BasicTokenizer()
    tokenizer.fit([sample_text])
    actual_vocab_size = tokenizer.vocab_size
    
    print(f"Vocabulary built with {actual_vocab_size} tokens")
    
    # Create model configuration
    model_config = AGI2Config(
        vocab_size=actual_vocab_size,
        n_positions=seq_len,
        n_ctx=seq_len
    )
    
    # Create model
    model = AGI2Model(model_config)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_path and Path(resume_path).exists():
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
            tokenizer=tokenizer,
            sources=sources, # Pass the list of sources
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seq_len=seq_len,
            device=device,
            save_path=f"trained/{model_name}",
            start_epoch=start_epoch
        )
        
        print(f"Training completed successfully!")
        print(f"Model saved to: trained/{model_name}.pt")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
