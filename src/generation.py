"""
Text Generation

This module provides text generation functions for the AGI2 model.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
import random

from .tokenizer import BasicTokenizer


def generate_text(
    model,
    prompt: str,
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    tokenizer = None,
    device: str = "cpu"
) -> str:
    """
    Generate text from a prompt using the AGI2 model.
    
    Args:
        prompt: Input text prompt
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        model: The trained AGI2 model
        tokenizer: Tokenizer for encoding/decoding
        
    Returns:
        Generated text string
    """
    if tokenizer is None:
        raise ValueError("Tokenizer is required for text generation")
    
    model = model.to(device)
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Generate tokens
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            outputs = model(generated_ids)
            next_token_logits = outputs[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                next_token_logits[top_k_indices] = top_k_logits
            
            # Apply top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the new token
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
            
            # Stop if we generate an end token
            if next_token.item() == tokenizer.vocab.get('<EOS>', -1):
                break
    
    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text


def generate_with_beam_search(
    model,
    prompt: str,
    max_length: int = 50,
    beam_width: int = 5,
    temperature: float = 1.0,
    tokenizer = None,
    device: str = "cpu"
) -> str:
    """
    Generate text using beam search for better quality.
    
    Args:
        prompt: Input text prompt
        max_length: Maximum length of generated text
        beam_width: Number of beams to maintain
        temperature: Sampling temperature
        model: The trained AGI2 model
        tokenizer: Tokenizer for encoding/decoding
        device: Device to run generation on
        
    Returns:
        Generated text string
    """
    if tokenizer is None:
        raise ValueError("Tokenizer is required for text generation")
    
    model = model.to(device)
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Initialize beams
    beams = [(input_ids.clone(), 0.0)]  # (sequence, score)
    
    with torch.no_grad():
        for _ in range(max_length):
            new_beams = []
            
            for beam_seq, beam_score in beams:
                # Get model predictions
                outputs = model(beam_seq)
                next_token_logits = outputs[0, -1, :]
                
                # Get top-k tokens
                top_k_logits, top_k_indices = torch.topk(next_token_logits, beam_width)
                
                for logit, token_id in zip(top_k_logits, top_k_indices):
                    new_seq = torch.cat([beam_seq, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = beam_score + logit.item()
                    new_beams.append((new_seq, new_score))
            
            # Keep top beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Check if all beams end with end token
            if all(beam[0][0, -1].item() == tokenizer.vocab.get('<EOS>', -1) for beam in beams):
                break
    
    # Decode the generated texts
    generated_texts = []
    for beam_seq, _ in beams:
        generated_text = tokenizer.decode(beam_seq[0].tolist())
        generated_texts.append(generated_text)
    
    return generated_texts


def generate_interactive(
    model,
    tokenizer,
    max_length: int = 100,
    temperature: float = 0.8,
    device: str = "cpu"
) -> None:
    """
    Interactive text generation loop.
    
    Args:
        model: The trained AGI2 model
        tokenizer: Tokenizer to use for encoding/decoding
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        device: Device to run generation on
    """
    print("Interactive text generation (type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        try:
            prompt = input("\nEnter your prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            print("\nGenerating...")
            generated_text = generate_text(
                model, prompt, max_length, temperature, tokenizer=tokenizer, device=device
            )
            
            print(f"\nGenerated text:\n{generated_text}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue
