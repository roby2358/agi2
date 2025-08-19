#!/usr/bin/env python3
"""
Memory Estimation Examples

This script demonstrates various memory estimation scenarios for AGI2 models.
Run it to see how different configurations affect GPU memory requirements.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import estimate_gpu_memory_requirements


def print_separator(title):
    """Print a formatted separator."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def estimate_configuration(name, **kwargs):
    """Estimate memory for a specific configuration."""
    print(f"\n{name}:")
    print(f"  Model: {kwargs['model_layer']} layers, {kwargs['model_embd']} dim, {kwargs['model_head']} heads")
    print(f"  Training: batch_size={kwargs['batch_size']}, seq_len={kwargs['seq_len']}")
    
    # Estimate for both precision levels
    memory_fp32 = estimate_gpu_memory_requirements(**kwargs, dtype_bytes=4)
    memory_fp16 = estimate_gpu_memory_requirements(**kwargs, dtype_bytes=2)
    
    print(f"  Parameters: {memory_fp16['total_params_millions']:.1f}M")
    print(f"  Memory: {memory_fp16['total_estimated_gb']:.2f} GB (FP16) / {memory_fp32['total_estimated_gb']:.2f} GB (FP32)")
    print(f"  Savings: {memory_fp32['total_estimated_gb'] - memory_fp16['total_estimated_gb']:.2f} GB")
    
    return memory_fp16


def main():
    """Run memory estimation examples."""
    print("AGI2 GPU Memory Estimation Examples")
    print("This script shows memory requirements for different model configurations")
    
    # Example 1: Small model (like lilwill.toml)
    print_separator("SMALL MODEL - Fast Training")
    small_config = {
        'model_positions': 512,
        'model_embd': 384,
        'model_layer': 6,
        'model_head': 6,
        'batch_size': 24,
        'seq_len': 512
    }
    small_memory = estimate_configuration("Small Model (lilwill.toml)", **small_config)
    
    # Example 2: Medium model
    print_separator("MEDIUM MODEL - Balanced")
    medium_config = {
        'model_positions': 512,
        'model_embd': 768,
        'model_layer': 12,
        'model_head': 12,
        'batch_size': 8,
        'seq_len': 512
    }
    medium_memory = estimate_configuration("Medium Model", **medium_config)
    
    # Example 3: Large model
    print_separator("LARGE MODEL - High Quality")
    large_config = {
        'model_positions': 1024,
        'model_embd': 1024,
        'model_layer': 24,
        'model_head': 16,
        'batch_size': 4,
        'seq_len': 512
    }
    large_memory = estimate_configuration("Large Model", **large_config)
    
    # Example 4: Very large model
    print_separator("VERY LARGE MODEL - Research Grade")
    xl_config = {
        'model_positions': 2048,
        'model_embd': 1536,
        'model_layer': 48,
        'model_head': 24,
        'batch_size': 1,
        'seq_len': 1024
    }
    xl_memory = estimate_configuration("Very Large Model", **xl_config)
    
    # Summary comparison
    print_separator("SUMMARY COMPARISON")
    print(f"{'Model Size':<20} {'Parameters':<12} {'Memory (FP16)':<15} {'GPU Requirement':<20}")
    print(f"{'-'*70}")
    
    models = [
        ("Small", small_memory),
        ("Medium", medium_memory),
        ("Large", large_memory),
        ("Very Large", xl_memory)
    ]
    
    for name, memory in models:
        params = f"{memory['total_params_millions']:.1f}M"
        mem_gb = f"{memory['total_estimated_gb']:.2f} GB"
        
        if memory['total_estimated_gb'] > 12:
            gpu_req = "24GB+ GPU"
        elif memory['total_estimated_gb'] > 8:
            gpu_req = "16GB+ GPU"
        elif memory['total_estimated_gb'] > 4:
            gpu_req = "8GB+ GPU"
        else:
            gpu_req = "Any GPU"
        
        print(f"{name:<20} {params:<12} {mem_gb:<15} {gpu_req:<20}")
    
    # Recommendations
    print_separator("RECOMMENDATIONS")
    print("Based on your GPU memory, here are recommended configurations:")
    
    print("\nFor 4GB GPU (GTX 1650, RTX 3050):")
    print("  ✅ Small model with batch_size=16")
    print("  ✅ Medium model with batch_size=4")
    print("  ❌ Large model (requires 8GB+)")
    
    print("\nFor 8GB GPU (RTX 3070, RTX 4060):")
    print("  ✅ Small model with batch_size=32")
    print("  ✅ Medium model with batch_size=16")
    print("  ✅ Large model with batch_size=4")
    print("  ❌ Very large model (requires 16GB+)")
    
    print("\nFor 16GB GPU (RTX 3080, RTX 4070 Ti):")
    print("  ✅ Any model size with reasonable batch sizes")
    print("  ✅ Very large model with batch_size=1")
    
    print("\nFor 24GB+ GPU (RTX 3090, RTX 4090, A100):")
    print("  ✅ Any model size with large batch sizes")
    print("  ✅ Multiple models simultaneously")
    
    print("\n💡 Tips:")
    print("  - Use mixed precision (FP16) to reduce memory by ~50%")
    print("  - Reduce batch_size if you run out of memory")
    print("  - Monitor actual usage with: nvidia-smi -l 1")
    print("  - Start with small models and scale up gradually")


if __name__ == '__main__':
    main()
