#!/usr/bin/env python3
"""
GPU Memory Estimation Tool

This script estimates GPU memory requirements for training AGI2 models
based on configuration files or command-line parameters.

Usage:
    python estimate_memory.py resources/lilwill.toml
    python estimate_memory.py --model-embd 384 --model-layer 6 --batch-size 24
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from utils import estimate_gpu_memory_requirements
    from config_loader import load_config
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def estimate_from_toml(toml_path: str) -> None:
    """Estimate memory requirements from a TOML configuration file."""
    try:
        config = load_config(toml_path)
        
        # Extract model parameters
        model_positions = config.get('model_positions', 512)
        model_embd = config.get('model_embd', 384)
        model_layer = config.get('model_layer', 6)
        model_head = config.get('model_head', 6)
        batch_size = config.get('batch_size', 4)
        seq_len = config.get('seq_len', 512)
        
        print(f"Configuration loaded from: {toml_path}")
        print(f"Model: {model_layer} layers, {model_embd} dim, {model_head} heads")
        print(f"Training: batch_size={batch_size}, seq_len={seq_len}")
        
        # Estimate memory for both float32 and float16
        memory_fp32 = estimate_gpu_memory_requirements(
            model_positions=model_positions,
            model_embd=model_embd,
            model_layer=model_layer,
            model_head=model_head,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype_bytes=4
        )
        
        memory_fp16 = estimate_gpu_memory_requirements(
            model_positions=model_positions,
            model_embd=model_embd,
            model_layer=model_layer,
            model_head=model_head,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype_bytes=2
        )
        
        print_memory_breakdown("Float32 (Standard Precision)", memory_fp32)
        print_memory_breakdown("Float16 (Mixed Precision)", memory_fp16)
        
        # Show memory savings
        savings = memory_fp32['total_estimated_gb'] - memory_fp16['total_estimated_gb']
        print(f"\nMixed Precision Memory Savings: {savings:.2f} GB")
        
        # Recommendations
        print_recommendations(memory_fp16['total_estimated_gb'], batch_size, seq_len)
        
    except Exception as e:
        print(f"Error loading configuration from {toml_path}: {e}")
        sys.exit(1)


def estimate_from_params(args) -> None:
    """Estimate memory requirements from command-line parameters."""
    memory_estimate = estimate_gpu_memory_requirements(
        model_positions=args.model_positions,
        model_embd=args.model_embd,
        model_layer=args.model_layer,
        model_head=args.model_head,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dtype_bytes=2 if args.mixed_precision else 4
    )
    
    print(f"Memory estimation for custom configuration:")
    print(f"Model: {args.model_layer} layers, {args.model_embd} dim, {args.model_head} heads")
    print(f"Training: batch_size={args.batch_size}, seq_len={args.seq_len}")
    print(f"Precision: {'Mixed (Float16)' if args.mixed_precision else 'Standard (Float32)'}")
    
    print_memory_breakdown("Memory Breakdown", memory_estimate)
    print_recommendations(memory_estimate['total_estimated_gb'], args.batch_size, args.seq_len)


def print_memory_breakdown(title: str, memory_estimate: dict) -> None:
    """Print a formatted memory breakdown."""
    print(f"\n{title}:")
    print(f"  Total parameters: {memory_estimate['total_params_millions']}M")
    print(f"  Estimated GPU memory: {memory_estimate['total_estimated_gb']:.2f} GB")
    print(f"  Breakdown:")
    print(f"    - Model weights: {memory_estimate['model_weights_gb']:.2f} GB")
    print(f"    - Activations: {memory_estimate['activations_gb']:.2f} GB")
    print(f"    - Optimizer state: {memory_estimate['optimizer_state_gb']:.2f} GB")
    print(f"    - Gradients: {memory_estimate['gradients_gb']:.2f} GB")
    print(f"    - IO tensors: {memory_estimate['io_tensors_gb']:.2f} GB")


def print_recommendations(total_memory_gb: float, batch_size: int, seq_len: int) -> None:
    """Print recommendations based on memory usage."""
    print(f"\nRecommendations:")
    
    if total_memory_gb > 12:
        print(f"  🚨 Very high memory usage ({total_memory_gb:.2f} GB)")
        print(f"     - Requires high-end GPU (24GB+) or multiple GPUs")
        print(f"     - Consider reducing model size or using gradient checkpointing")
    elif total_memory_gb > 8:
        print(f"  ⚠️  High memory usage ({total_memory_gb:.2f} GB)")
        print(f"     - Requires high-end GPU (16GB+)")
        print(f"     - Consider reducing batch_size or seq_len")
    elif total_memory_gb > 4:
        print(f"  ⚠️  Moderate memory usage ({total_memory_gb:.2f} GB)")
        print(f"     - Requires mid-range GPU (8GB+)")
        print(f"     - Monitor GPU memory during training")
    else:
        print(f"  ✅ Memory usage looks reasonable ({total_memory_gb:.2f} GB)")
        print(f"     - Should work on most modern GPUs")
    
    # Specific suggestions
    if batch_size > 16:
        print(f"  💡 Consider reducing batch_size from {batch_size} to 8-16 for lower memory usage")
    if seq_len > 512:
        print(f"  💡 Consider reducing seq_len from {seq_len} to 256-512 for lower memory usage")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate GPU memory requirements for AGI2 model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python estimate_memory.py resources/lilwill.toml
  python estimate_memory.py --model-embd 384 --model-layer 6 --batch-size 24
  python estimate_memory.py --model-embd 768 --model-layer 12 --batch-size 8 --mixed-precision
        """
    )
    
    # File-based estimation
    parser.add_argument('toml_file', nargs='?', help='Path to TOML configuration file')
    
    # Parameter-based estimation
    parser.add_argument('--model-embd', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--model-layer', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--model-head', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--model-positions', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    parser.add_argument('--seq-len', type=int, default=512, help='Training sequence length')
    parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision (float16)')
    
    args = parser.parse_args()
    
    if args.toml_file:
        # Estimate from TOML file
        if not Path(args.toml_file).exists():
            print(f"Error: Configuration file '{args.toml_file}' not found")
            sys.exit(1)
        estimate_from_toml(args.toml_file)
    else:
        # Estimate from command-line parameters
        estimate_from_params(args)


if __name__ == '__main__':
    main()
