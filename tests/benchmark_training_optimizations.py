"""
Benchmark Training Optimizations

This script benchmarks the performance improvements from the training optimizations.
"""

import torch
import time
import gc
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import AGI2Model
from src.config import AGI2Config
from src.training import train_epoch


def benchmark_causal_mask_caching():
    """Benchmark the causal mask caching optimization."""
    print("=== Benchmarking Causal Mask Caching ===")
    
    # Create a small model for testing
    config = AGI2Config(
        vocab_size=1000,
        n_positions=512,
        n_embd=128,
        n_layer=2,
        n_head=4,
        n_inner=256,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        tie_word_embeddings=False
    )
    
    model = AGI2Model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    seq_len = 512
    num_iterations = 1000
    
    # Benchmark without caching (simulate old behavior)
    print(f"Running {num_iterations} forward passes with sequence length {seq_len}...")
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = model(torch.randint(0, 1000, (1, seq_len), device=device))
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(torch.randint(0, 1000, (1, seq_len), device=device))
    
    total_time = time.time() - start_time
    avg_time = total_time / num_iterations
    
    print(f"Average forward pass time: {avg_time*1000:.2f}ms")
    print(f"Total time: {total_time:.2f}s")
    print(f"Cache size: {len(model._causal_mask_cache)}")
    
    # Show cache contents
    for (cached_seq_len, cached_device), mask in model._causal_mask_cache.items():
        print(f"  Cached mask: seq_len={cached_seq_len}, device={cached_device}, shape={mask.shape}")
    
    return avg_time


def benchmark_memory_usage():
    """Benchmark memory usage with and without optimizations."""
    print("\n=== Benchmarking Memory Usage ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory benchmark")
        return
    
    config = AGI2Config(
        vocab_size=1000,
        n_positions=512,
        n_embd=128,
        n_layer=2,
        n_head=4,
        n_inner=256,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        tie_word_embeddings=False
    )
    
    model = AGI2Model(config)
    device = torch.device('cuda')
    model = model.to(device)
    
    # Clear cache and measure baseline memory
    torch.cuda.empty_cache()
    gc.collect()
    
    baseline_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    baseline_reserved = torch.cuda.memory_reserved() / 1024**2    # MB
    
    print(f"Baseline memory: {baseline_allocated:.1f}MB allocated, {baseline_reserved:.1f}MB reserved")
    
    # Run some forward passes to populate cache
    seq_len = 512
    for _ in range(10):
        with torch.no_grad():
            _ = model(torch.randint(0, 1000, (1, seq_len), device=device))
    
    # Measure memory after caching
    cached_allocated = torch.cuda.memory_allocated() / 1024**2
    cached_reserved = torch.cuda.memory_reserved() / 1024**2
    
    print(f"After caching: {cached_allocated:.1f}MB allocated, {cached_reserved:.1f}MB reserved")
    print(f"Cache overhead: {cached_allocated - baseline_allocated:.1f}MB")
    
    # Clear cache and measure memory
    model.clear_mask_cache()
    torch.cuda.empty_cache()
    gc.collect()
    
    cleared_allocated = torch.cuda.memory_allocated() / 1024**2
    cleared_reserved = torch.cuda.memory_reserved() / 1024**2
    
    print(f"After clearing cache: {cleared_allocated:.1f}MB allocated, {cleared_reserved:.1f}MB reserved")
    print(f"Memory recovered: {cached_allocated - cleared_allocated:.1f}MB")


def benchmark_amp_performance():
    """Benchmark AMP performance if available."""
    print("\n=== Benchmarking AMP Performance ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping AMP benchmark")
        return
    
    config = AGI2Config(
        vocab_size=1000,
        n_positions=512,
        n_embd=128,
        n_layer=2,
        n_head=4,
        n_inner=256,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        tie_word_embeddings=False
    )
    
    model = AGI2Model(config)
    device = torch.device('cuda')
    model = model.to(device)
    
    # Create mock training components
    batch_size = 4
    seq_len = 512
    
    # Mock dataloader
    class MockDataLoader:
        def __init__(self, num_batches):
            self.num_batches = num_batches
        
        def __len__(self):
            return self.num_batches
        
        def __iter__(self):
            for _ in range(self.num_batches):
                # Return tensor directly, not tuple
                yield torch.randint(0, 1000, (batch_size, seq_len), device=device)
    
    dataloader = MockDataLoader(100)
    
    # Mock optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Benchmark with AMP disabled
    print("Benchmarking without AMP...")
    torch.cuda.empty_cache()
    gc.collect()
    
    start_time = time.time()
    avg_loss_no_amp = train_epoch(
        model, dataloader, optimizer, criterion, device,
        use_amp=False, log_gpu_memory=False
    )
    time_no_amp = time.time() - start_time
    
    print(f"Without AMP: {time_no_amp:.2f}s, Loss: {avg_loss_no_amp:.4f}")
    
    # Benchmark with AMP enabled
    print("Benchmarking with AMP...")
    torch.cuda.empty_cache()
    gc.collect()
    
    start_time = time.time()
    avg_loss_with_amp = train_epoch(
        model, dataloader, optimizer, criterion, device,
        use_amp=True, log_gpu_memory=False
    )
    time_with_amp = time.time() - start_time
    
    print(f"With AMP: {time_with_amp:.2f}s, Loss: {avg_loss_with_amp:.4f}")
    
    if time_no_amp > 0:
        speedup = time_no_amp / time_with_amp
        print(f"AMP speedup: {speedup:.2f}x")


def main():
    """Run all benchmarks."""
    print("Training Optimizations Benchmark")
    print("=" * 50)
    
    try:
        # Run benchmarks
        benchmark_causal_mask_caching()
        benchmark_memory_usage()
        benchmark_amp_performance()
        
        print("\n" + "=" * 50)
        print("Benchmark completed!")
        
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
