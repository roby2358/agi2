#!/usr/bin/env python3
"""
AGI2 Beam Search Generation Script

This script generates text using beam search with a trained AGI2 model.
Usage: python agi2_generate_beam.py <config_file> <prompt>
"""

import sys
import torch
from pathlib import Path


from src.config_loader import get_generation_config, get_config_value
from src.cuda_utils import check_cuda_availability, get_optimal_device
from src.config import AGI2Config
from src.model import AGI2Model
from src.basic_tokenizer import BasicTokenizer
from src.generation import generate_with_beam_search








def main():
    if len(sys.argv) < 2:
        print("Usage: python agi2_generate_beam.py <config_file> [prompt]")
        print("Example: python agi2_generate_beam.py resources/moby_dick.toml \"The future of AI is\"")
        sys.exit(1)
    
    config_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "The future of AI is"
    
    try:
        # Load configuration from TOML file
        config = get_generation_config(config_path)
        print(f"Loaded configuration from: {config_path}")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Check CUDA availability at startup
    print("Checking CUDA availability for beam search generation...")
    cuda_status = check_cuda_availability(verbose=True)
    
    # Extract configuration values with defaults
    model_path = Path(get_config_value(config, 'model_path'))
    max_length = get_config_value(config, 'max_length', 50)
    beam_size = get_config_value(config, 'beam_size', 5)
    device_choice = get_config_value(config, 'device', 'auto')
    model_seed = get_config_value(config, 'model_seed', "")
    
    # Validate model path
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print("Please train a model first using agi2_train.py")
        sys.exit(1)
    
    # Determine device using our utility
    device = get_optimal_device(device_choice)
    
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")
    print(f"User prompt: '{prompt}'")
    if model_seed:
        print(f"Model seed: {len(model_seed)} characters")
        print(f"Full context: {len(model_seed + prompt)} characters")
    print(f"Max length: {max_length}")
    print(f"Beam size: {beam_size}")
    print("-" * 50)
    
    try:
        # Load the trained model
        model = torch.load(str(model_path), map_location=device)
        model.eval()
        
        # Initialize tokenizer
        tokenizer = BasicTokenizer()
        
        # Combine model seed with user prompt
        full_prompt = model_seed + prompt if model_seed else prompt
        
        # Generate text using beam search
        generated_text = generate_with_beam_search(
            model=model,
            prompt=full_prompt,
            max_length=max_length,
            beam_size=beam_size,
            tokenizer=tokenizer
        )
        
        print("Generated text (beam search):")
        print(generated_text)
        
    except Exception as e:
        print(f"Generation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
