# Training Optimizations

This document outlines the performance optimizations implemented in the AGI2 training pipeline to reduce redundant work, improve GPU utilization, and speed up training.

## Overview of Optimizations

### 1. Causal Mask Caching ⚡
**Problem**: The causal attention mask was being recreated on every forward pass, even though it only depends on sequence length and device.

**Solution**: Implemented a caching system that stores masks per `(sequence_length, device)` combination.

**Impact**: 
- Eliminates redundant tensor creation for repeated sequence lengths
- Significant speedup for models with fixed or repeated sequence lengths
- Memory overhead is minimal (typically <1MB for common sequence lengths)

**Implementation**: 
- Added `_causal_mask_cache` dictionary to `AGI2Model`
- Modified `_create_causal_mask()` to check cache first
- Added `clear_mask_cache()` method for memory management

### 2. Automatic Mixed Precision (AMP) 🚀
**Problem**: Training was using full precision (FP32) throughout, missing potential speedups on modern GPUs.

**Solution**: Added AMP support with `torch.cuda.amp` for faster training and reduced memory usage.

**Impact**:
- **1.5x to 2x speedup** on modern NVIDIA GPUs (RTX 3000/4000 series, A100, H100)
- Reduced GPU memory usage by ~25-30%
- Maintains training accuracy

**Implementation**:
- Added `use_amp` parameter to training functions
- Integrated `GradScaler` for stable mixed precision training
- Automatic fallback to FP32 if AMP fails

### 3. Optimized Data Transfer 📊
**Problem**: Data transfer to GPU was not optimized, potentially causing bottlenecks.

**Solution**: Added `non_blocking=True` for asynchronous data transfer and DataLoader optimizations.

**Impact**:
- Faster data transfer to GPU
- Better GPU utilization during training
- Reduced CPU-GPU synchronization overhead

**Implementation**:
- Added `non_blocking=True` to `batch.to(device)`
- Added `pin_memory=True` for faster CPU-GPU transfer
- Added `num_workers` support for parallel data loading
- Added `persistent_workers` for worker process reuse

### 4. Reduced GPU Memory Queries 🔍
**Problem**: GPU memory information was queried multiple times per epoch, adding unnecessary overhead.

**Solution**: Made GPU memory logging optional and reduced query frequency.

**Impact**:
- Reduced overhead from repeated CUDA API calls
- Better performance when memory monitoring isn't needed
- Configurable logging for debugging vs. production

**Implementation**:
- Added `log_gpu_memory` parameter to control logging
- Cached GPU memory info per epoch instead of per batch
- Only query memory when explicitly requested

## Performance Improvements

### Expected Speedups
- **Causal Mask Caching**: 5-15% faster forward passes for repeated sequence lengths
- **AMP**: 1.5x to 2x overall training speedup on modern GPUs
- **Data Transfer Optimization**: 10-20% faster data loading
- **Combined**: **2x to 3x overall training speedup** on modern hardware

### Memory Usage
- **AMP**: 25-30% reduction in GPU memory usage
- **Causal Mask Cache**: Minimal overhead (<1MB for typical use cases)
- **DataLoader Optimization**: Better memory management and reduced fragmentation

## Usage Examples

### Basic Training with Optimizations
```python
from src.training import train_model

# Training with all optimizations enabled (default)
history = train_model(
    model=model,
    tokenizer=tokenizer,
    sources="path/to/corpus.txt",
    epochs=10,
    batch_size=8,
    use_amp=True,           # Enable AMP (default)
    log_gpu_memory=False,   # Disable memory logging for production
    num_workers=4,          # Use 4 worker processes
    pin_memory=True         # Enable memory pinning
)
```

### Training without AMP (for compatibility)
```python
# Training without AMP for older GPUs or debugging
history = train_model(
    model=model,
    tokenizer=tokenizer,
    sources="path/to/corpus.txt",
    epochs=10,
    batch_size=8,
    use_amp=False,          # Disable AMP
    log_gpu_memory=True     # Enable memory logging for debugging
)
```

### Memory Management
```python
# Clear mask cache if needed (e.g., switching devices)
model.clear_mask_cache()

# Or manually clear specific masks
model._causal_mask_cache.clear()
```

## Configuration Options

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_amp` | `True` | Enable automatic mixed precision |
| `log_gpu_memory` | `False` | Log GPU memory usage during training |
| `num_workers` | `0` | Number of DataLoader worker processes |
| `pin_memory` | `True` | Pin memory for faster GPU transfer |

### Model Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `_causal_mask_cache` | `{}` | Internal cache for attention masks |
| `clear_mask_cache()` | N/A | Method to clear mask cache |

## Testing

### Unit Tests
Run the optimization tests:
```bash
cd tests
python -m pytest test_training_optimizations.py -v
```

### Performance Benchmarks
Run the performance benchmarks:
```bash
cd tests
python benchmark_training_optimizations.py
```

## Compatibility

### GPU Requirements
- **AMP**: Requires CUDA-compatible GPU with compute capability 7.0+
- **Causal Mask Caching**: Works on all devices (CPU/GPU)
- **Data Transfer Optimization**: Works on all devices, optimized for GPU

### PyTorch Version
- **Minimum**: PyTorch 1.7.0+ (for AMP support)
- **Recommended**: PyTorch 2.0.0+ (for best performance)

## Future Optimizations

### Planned Improvements
1. **Gradient Checkpointing**: For training larger models with limited memory
2. **Model Sharding**: For distributed training across multiple GPUs
3. **Dynamic Batching**: Adaptive batch sizes based on memory availability
4. **Attention Optimization**: Flash Attention or other optimized attention implementations

### Research Areas
1. **Memory-Efficient Attention**: Reducing memory usage for long sequences
2. **Adaptive Precision**: Dynamic precision adjustment based on layer importance
3. **Compression**: Model compression techniques for faster inference

## Troubleshooting

### Common Issues

#### AMP Errors
```python
# If AMP causes issues, disable it
history = train_model(..., use_amp=False)
```

#### Memory Issues
```python
# Clear mask cache if memory usage is high
model.clear_mask_cache()

# Reduce batch size or sequence length
history = train_model(..., batch_size=4, seq_len=256)
```

#### Performance Issues
```python
# Ensure CUDA is properly configured
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Check if optimizations are working
print(f"Model cache size: {len(model._causal_mask_cache)}")
```

## Conclusion

These optimizations provide significant performance improvements while maintaining training stability and accuracy. The causal mask caching eliminates redundant work, AMP provides substantial speedups on modern hardware, and the data transfer optimizations reduce bottlenecks in the training pipeline.

For best results:
1. Use AMP on modern NVIDIA GPUs
2. Adjust `num_workers` based on your system (typically 2-8)
3. Monitor memory usage and clear cache if needed
4. Test optimizations on your specific hardware and dataset

The optimizations are designed to be backward compatible and can be disabled if needed for debugging or compatibility reasons.
