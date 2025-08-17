#!/usr/bin/env python3
"""
AGI2 Text Generation Script

Usage:
    python agi2_generate.py <prompt>

Example:
    python agi2_generate.py "Once upon a time"
"""

import sys
import argparse
import torch
from pathlib import Path

from src.generation import generate_text
from src.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained AGI2 model")
    parser.add_argument(
        "prompt", 
        type=str, 
        help="Text prompt to start generation"
    )
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="trained_model.pth", 
        help="Path to the trained model file (default: trained_model.pth)"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=100, 
        help="Maximum length of generated text (default: 100)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.8, 
        help="Sampling temperature (default: 0.8)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cpu", "cuda", "auto"], 
        default="auto", 
        help="Device to use for generation (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print("Please train a model first using agi2_train.py")
        sys.exit(1)
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Max length: {args.max_length}")
    print(f"Temperature: {args.temperature}")
    print("-" * 50)
    
    try:
        # Load the trained model
        model = torch.load(str(model_path), map_location=device)
        model.eval()
        
        # Initialize tokenizer
        tokenizer = Tokenizer()
        
        # Generate text
        generated_text = generate_text(
            model=model,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            tokenizer=tokenizer
        )
        
        print("Generated text:")
        print(generated_text)
        
    except Exception as e:
        print(f"Generation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
