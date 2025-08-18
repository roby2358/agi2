#!/usr/bin/env python3
"""
AGI2 Text Generation Script

This script generates text using a trained AGI2 model.
Usage: python agi2_generate.py <config_file> <prompt>
"""

import sys
import torch
from pathlib import Path


from src.config_loader import load_config, get_config_value
from src.cuda_utils import check_cuda_availability, get_optimal_device
from src.config import AGI2Config
from src.model import AGI2Model
from src.tokenizer import BasicTokenizer
from src.generation import generate_text








def main():
    if len(sys.argv) < 2:
        print("Usage: python agi2_generate.py <config_file> [prompt]")
        print("Example: python agi2_generate.py resources/moby_dick.toml \"Call me Ishmael\"")
        sys.exit(1)
    
    config_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Once upon a time"
    
    try:
        # Load configuration from TOML file
        config = load_config(config_path)
        print(f"Loaded configuration from: {config_path}")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Check CUDA availability at startup
    print("Checking CUDA availability for text generation...")
    cuda_status = check_cuda_availability(verbose=True)
    
    # Extract configuration values with defaults
    model_path = Path(get_config_value(config, 'model_path'))
    max_length = get_config_value(config, 'max_length', 100)
    temperature = get_config_value(config, 'temperature', 0.8)
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
    print(f"Temperature: {temperature}")
    print("-" * 50)
    
    try:
        # Load the trained model checkpoint
        checkpoint = torch.load(str(model_path), map_location=device)
        
        # Check if it's a checkpoint dictionary or direct model
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # It's a checkpoint dictionary
            print("Loading from checkpoint dictionary...")
            
            # Extract config and create model
            if 'config' in checkpoint:
                config_obj = checkpoint['config']
                print(f"Model config: {config_obj}")
            else:
                # Fallback to default config if not saved
                print("No config found in checkpoint, using default AGI2 Small config")
                config_obj = AGI2Config()
            
            # Create new model instance
            model = AGI2Model(config_obj)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load tokenizer if available
            if 'tokenizer' in checkpoint:
                tokenizer = checkpoint['tokenizer']
                print("Using tokenizer from checkpoint")
            else:
                tokenizer = BasicTokenizer()
                print("Using default tokenizer")
                
        else:
            # Assume it's a direct model
            print("Loading direct model...")
            model = checkpoint
            tokenizer = BasicTokenizer()
        
        model = model.to(device)
        model.eval()
        
        # Combine model seed with user prompt
        full_prompt = model_seed + prompt if model_seed else prompt
        
        # Generate text
        generated_text = generate_text(
            model=model,
            prompt=full_prompt,
            max_length=max_length,
            temperature=temperature,
            tokenizer=tokenizer
        )
        
        print("Generated text:")
        print(generated_text)
        
    except Exception as e:
        print(f"Generation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
