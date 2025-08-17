#!/usr/bin/env python3
"""
AGI2 Training Script

Usage:
    python agi2_train.py <corpus_path>

Example:
    python agi2_train.py data/corpus.txt
    python agi2_train.py data/corpus.txt --resume trained/model.pt_epoch_10.pt
    python agi2_train.py data/corpus.txt --model-name my_model
"""

import sys
import argparse
import torch
from pathlib import Path

from src.model import GPT2Model
from src.tokenizer import BasicTokenizer
from src.training import train_model
from src.config import GPT2Config
from src.utils import load_checkpoint
from src.cuda_utils import check_cuda_availability, get_optimal_device


def main():
    # Check CUDA availability at startup
    print("Checking CUDA availability for training...")
    cuda_status = check_cuda_availability(verbose=True)
    
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
        "--model-name", 
        type=str, 
        default="model", 
        help="Name for the model (default: model) - files will be saved to trained/{model-name}.pt and trained/{model-name}.pt_epoch_N.pt"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        help="Path to checkpoint file to resume training from"
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
    
    # Validate resume checkpoint if provided
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"Error: Resume checkpoint not found: {resume_path}")
            sys.exit(1)
        print(f"Resuming training from checkpoint: {resume_path}")
    
    # Determine device using our utility
    device = get_optimal_device(args.device)
    
    print(f"Using device: {device}")
    print(f"Training on corpus: {corpus_path}")
    
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
    config = GPT2Config(
        vocab_size=actual_vocab_size,  # Use actual vocab size from tokenizer
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12
    )
    
    model = GPT2Model(config)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        
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
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seq_len=args.seq_len,
            device=device,
            save_path=args.model_name,
            start_epoch=start_epoch  # Pass starting epoch
        )
        
        print(f"Training completed successfully!")
        print(f"Model saved to: trained/{args.model_name}.pt")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
