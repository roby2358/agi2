# GPU Memory Estimation Guide

This guide explains how to estimate GPU memory requirements for training AGI2 models using the built-in memory estimation tools.

## Overview

The AGI2 project now includes comprehensive GPU memory estimation functionality that can:

1. **Estimate memory requirements** before training starts
2. **Compare different precision levels** (float32 vs float16)
3. **Provide recommendations** based on memory usage
4. **Work with TOML configuration files** or custom parameters

## How It Works

The memory estimation calculates requirements for:

- **Model weights**: The actual neural network parameters
- **Activations**: Intermediate values during forward pass
- **Optimizer state**: AdamW momentum and variance (2x parameters)
- **Gradients**: Gradients during backward pass
- **IO tensors**: Input/output tensors and buffers

## Usage

### 1. Automatic Estimation During Training

When you start training, the system automatically estimates GPU memory requirements and displays them along with the nvidia monitoring commands:

```bash
python agi2_train.py resources/lilwill.toml
```

You'll see output like:
```
GPU Memory Requirements Estimate:
  Model parameters: 30.12M
  Estimated GPU memory: 0.31 GB
  Breakdown:
    - Model weights: 0.06 GB
    - Activations: 0.08 GB
    - Optimizer state: 0.11 GB
    - Gradients: 0.06 GB
    - IO tensors: 0.01 GB
  
  ✅ Memory usage looks reasonable
```

### 2. Standalone Memory Estimation Tool

Use the dedicated script to estimate memory requirements without starting training:

```bash
# Estimate from TOML configuration file
python estimate_memory.py resources/lilwill.toml

# Estimate with custom parameters
python estimate_memory.py --model-embd 768 --model-layer 12 --batch-size 8 --mixed-precision
```

### 3. Programmatic Usage

You can also use the memory estimation function directly in your code:

```python
from src.utils import estimate_gpu_memory_requirements

memory_estimate = estimate_gpu_memory_requirements(
    model_positions=512,
    model_embd=384,
    model_layer=6,
    model_head=6,
    batch_size=24,
    seq_len=512,
    dtype_bytes=2,  # 2 for float16, 4 for float32
    include_activations=True,
    include_optimizer=True
)

print(f"Total estimated memory: {memory_estimate['total_estimated_gb']:.2f} GB")
```

## Examples

### Example 1: Small Model (lilwill.toml)

```bash
python estimate_memory.py resources/lilwill.toml
```

**Configuration:**
- Model: 6 layers, 384 dim, 6 heads
- Training: batch_size=24, seq_len=512
- Parameters: 30.12M

**Memory Requirements:**
- Float32: 0.54 GB
- Float16: 0.31 GB
- **Mixed Precision Savings: 0.23 GB**

**Recommendation:** ✅ Memory usage looks reasonable

### Example 2: Medium Model

```bash
python estimate_memory.py --model-embd 768 --model-layer 12 --batch-size 8 --mixed-precision
```

**Configuration:**
- Model: 12 layers, 768 dim, 6 heads
- Training: batch_size=8, seq_len=512
- Parameters: 123.94M

**Memory Requirements:**
- Float16: 1.03 GB

**Recommendation:** ✅ Memory usage looks reasonable

### Example 3: Large Model

```bash
python estimate_memory.py --model-embd 1024 --model-layer 24 --batch-size 4 --mixed-precision
```

**Configuration:**
- Model: 24 layers, 1024 dim, 16 heads
- Training: batch_size=4, seq_len=512
- Parameters: 442.37M

**Memory Requirements:**
- Float16: 3.67 GB

**Recommendation:** ⚠️ Moderate memory usage - monitor GPU memory during training

## Memory Breakdown

The estimation provides detailed breakdown of memory usage:

- **Model weights**: Core neural network parameters
- **Activations**: Intermediate values stored during forward pass
- **Optimizer state**: AdamW momentum and variance (typically 2x parameters)
- **Gradients**: Gradients computed during backward pass
- **IO tensors**: Input/output tensors and temporary buffers

## Precision Impact

**Float32 (Standard Precision):**
- Higher memory usage
- Better numerical stability
- Slower training

**Float16 (Mixed Precision):**
- Lower memory usage (~50% reduction)
- Faster training
- Requires careful implementation for stability

## Recommendations

The system provides automatic recommendations based on memory usage:

- **< 4 GB**: ✅ Memory usage looks reasonable
- **4-8 GB**: ⚠️ Moderate memory usage - monitor GPU memory during training
- **8-12 GB**: ⚠️ High memory usage - consider reducing batch_size or seq_len
- **> 12 GB**: 🚨 Very high memory usage - requires high-end GPU or multiple GPUs

## Tips for Reducing Memory Usage

1. **Reduce batch size**: Smaller batches use less memory
2. **Use mixed precision**: Float16 can reduce memory by ~50%
3. **Reduce sequence length**: Shorter sequences use less activation memory
4. **Use gradient checkpointing**: Trade computation for memory (future feature)
5. **Reduce model size**: Fewer layers or smaller embedding dimensions

## GPU Memory Monitoring

During training, use these commands to monitor actual GPU memory usage:

```bash
# Watch GPU status every 1 second
nvidia-smi -l 1

# Watch only memory usage
nvidia-smi -l 1 --query-gpu=memory.used,memory.total,memory.free --format=csv
```

## Testing

Run the memory estimation tests to verify functionality:

```bash
python tests/test_memory_estimation.py
```

## Limitations

- **Estimates only**: Actual memory usage may vary based on implementation details
- **PyTorch specific**: Assumes PyTorch memory management patterns
- **Simplified activations**: Activation memory is estimated, not precisely calculated
- **No gradient checkpointing**: Assumes standard backpropagation

## Future Enhancements

- **Gradient checkpointing support**: More accurate activation memory estimation
- **Memory profiling**: Integration with PyTorch memory profilers
- **Multi-GPU support**: Distributed training memory estimation
- **Dynamic batching**: Adaptive batch size based on available memory

## Troubleshooting

**Import errors**: Make sure you're running from the project root directory
**Configuration errors**: Verify TOML file format and parameter names
**Memory mismatch**: Compare estimates with actual `nvidia-smi` output

## Conclusion

The GPU memory estimation tools provide valuable insights into training requirements before you start. Use them to:

1. **Plan your training setup** based on available GPU memory
2. **Optimize parameters** for your hardware constraints
3. **Compare configurations** to find the best balance of model size and training efficiency
4. **Monitor training** with the provided nvidia commands

This helps ensure successful training runs and efficient use of your GPU resources.
