#!/usr/bin/env python3
"""
AGI2 Beam Search Text Generation Script

Usage:
    python agi2_generate_beam.py <prompt>

Example:
    python agi2_generate_beam.py "The future of AI is"
"""

import sys
import argparse
import torch
from pathlib import Path

from src.model import GPT2Model
from src.tokenizer import BasicTokenizer
from src.generation import generate_with_beam_search
from src.config import GPT2Config
from src.cuda_utils import check_cuda_availability, get_optimal_device


def main():
    # Check CUDA availability at startup
    print("Checking CUDA availability for beam search generation...")
    cuda_status = check_cuda_availability(verbose=True)
    
    parser = argparse.ArgumentParser(description="Generate text using beam search with a trained AGI2 model")
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
        default=50, 
        help="Maximum length of generated text (default: 50)"
    )
    parser.add_argument(
        "--beam-size", 
        type=int, 
        default=5, 
        help="Beam size for search (default: 5)"
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
    
    # Determine device using our utility
    device = get_optimal_device(args.device)
    
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Max length: {args.max_length}")
    print(f"Beam size: {args.beam_size}")
    print("-" * 50)
    
    try:
        # Load the trained model
        model = torch.load(str(model_path), map_location=device)
        model.eval()
        
        # Initialize tokenizer
        tokenizer = BasicTokenizer()
        
        # Generate text using beam search
        generated_text = generate_with_beam_search(
            model=model,
            prompt=args.prompt,
            max_length=args.max_length,
            beam_size=args.beam_size,
            tokenizer=tokenizer
        )
        
        print("Generated text (beam search):")
        print(generated_text)
        
    except Exception as e:
        print(f"Generation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
