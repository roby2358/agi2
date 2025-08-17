#!/usr/bin/env python3
"""
AGI2 Interactive Chat Script

Usage:
    python agi2_interactive.py

Example:
    python agi2_interactive.py
"""

import sys
import torch
from pathlib import Path

from src.interactive import InteractivePrompt
from src.tokenizer import Tokenizer


def main():
    # Default model path
    model_path = Path("trained_model.pth")
    
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print("Please train a model first using agi2_train.py")
        sys.exit(1)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")
    
    try:
        # Load the trained model
        model = torch.load(str(model_path), map_location=device)
        model.eval()
        
        # Initialize tokenizer
        tokenizer = Tokenizer()
        
        # Create interactive session
        chat = InteractivePrompt(
            model=model,
            max_context_length=1024,
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
