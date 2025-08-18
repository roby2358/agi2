#!/usr/bin/env python3
"""
AGI2 Interactive Generation Script

This script provides an interactive interface for text generation with a trained AGI2 model.
Usage: python agi2_interactive.py <config_file>
"""

import sys
import torch
from pathlib import Path


from src.config_loader import get_interactive_config, get_config_value
from src.cuda_utils import check_cuda_availability, get_optimal_device
from src.config import AGI2Config
from src.model import AGI2Model
from src.tokenizer import BasicTokenizer
from src.generation import generate_interactive








def main():
    if len(sys.argv) != 2:
        print("Usage: python agi2_interactive.py <config_file>")
        print("Example: python agi2_interactive.py resources/moby_dick.toml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    try:
        # Load configuration from TOML file
        config = get_interactive_config(config_path)
        print(f"Loaded configuration from: {config_path}")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Check CUDA availability at startup
    print("Checking CUDA availability for interactive chat...")
    cuda_status = check_cuda_availability(verbose=True)
    
    # Extract configuration values with defaults
    model_path = Path(get_config_value(config, 'model_path'))
    device_choice = get_config_value(config, 'device', 'auto')
    max_context_length = get_config_value(config, 'max_context_length', 1024)
    
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print("Please train a model first using agi2_train.py")
        sys.exit(1)
    
    # Determine device using our utility
    device = get_optimal_device(device_choice)
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")
    
    try:
        # Load the trained model
        model = torch.load(str(model_path), map_location=device)
        model.eval()
        
        # Initialize tokenizer
        tokenizer = BasicTokenizer()
        
        # Create interactive session
        chat = generate_interactive(
            model=model,
            max_context_length=max_context_length,
            tokenizer=tokenizer
        )
        
        print("Interactive chat session started!")
        print("Type 'quit' or 'exit' to end the session")
        print("-" * 50)
        
        # Start chatting
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Get response from model
                response = chat.send_message(user_input)
                print(f"AI: {response}")
                print()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break
        
    except Exception as e:
        print(f"Interactive session failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
