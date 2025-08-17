#!/usr/bin/env python3
"""
AGI2 Training Script

Usage:
    python agi2_train.py <corpus_path>

Example:
    python agi2_train.py data/corpus.txt
"""

import sys
import argparse
import torch
from pathlib import Path

from src.model import GPT2Model
from src.tokenizer import Tokenizer
from src.training import train_model
from src.config import ModelConfig


def main():
    parser = argparse.ArgumentParser(description="Train AGI2 model on a corpus")
    parser.add_argument(
        "corpus_path", 
        type=str, 
        help="Path to the training corpus file"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10, 
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=4, 
        help="Training batch size (default: 4)"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=3e-4, 
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--seq-len", 
        type=int, 
        default=1024, 
        help="Sequence length (default: 1024)"
    )
    parser.add_argument(
        "--save-path", 
        type=str, 
        default="trained_model.pth", 
        help="Path to save the trained model (default: trained_model.pth)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cpu", "cuda", "auto"], 
        default="auto", 
        help="Device to use for training (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Validate corpus path
    corpus_path = Path(args.corpus_path)
    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {corpus_path}")
        sys.exit(1)
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Training on corpus: {corpus_path}")
    
    # Initialize model and tokenizer
    config = ModelConfig(
        vocab_size=50000,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12
    )
    
    model = GPT2Model(config)
    tokenizer = Tokenizer()
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train the model
    try:
        training_history = train_model(
            model=model,
            corpus_path=str(corpus_path),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seq_len=args.seq_len,
            device=device,
            save_path=args.save_path
        )
        
        print(f"Training completed successfully!")
        print(f"Model saved to: {args.save_path}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
