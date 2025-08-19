"""
Test GPU Memory Estimation

This module tests the GPU memory estimation functionality.
"""

import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import estimate_gpu_memory_requirements


class TestMemoryEstimation(unittest.TestCase):
    """Test cases for GPU memory estimation."""
    
    def test_lilwill_config_memory_estimation(self):
        """Test memory estimation for lilwill.toml configuration."""
        # Parameters from lilwill.toml
        model_positions = 512
        model_embd = 384
        model_layer = 6
        model_head = 6
        batch_size = 24
        seq_len = 512
        
        # Estimate memory requirements
        memory_estimate = estimate_gpu_memory_requirements(
            model_positions=model_positions,
            model_embd=model_embd,
            model_layer=model_layer,
            model_head=model_head,
            batch_size=batch_size,
            seq_len=seq_len
        )
        
        # Print the results
        print(f"\nGPU Memory Estimation for lilwill.toml configuration:")
        print(f"Model: {model_layer} layers, {model_embd} dim, {model_head} heads")
        print(f"Training: batch_size={batch_size}, seq_len={seq_len}")
        print(f"Total parameters: {memory_estimate['total_params_millions']}M")
        print(f"Model weights: {memory_estimate['model_weights_gb']} GB")
        print(f"Activations: {memory_estimate['activations_gb']} GB")
        print(f"Optimizer state: {memory_estimate['optimizer_state_gb']} GB")
        print(f"Gradients: {memory_estimate['gradients_gb']} GB")
        print(f"IO tensors: {memory_estimate['io_tensors_gb']} GB")
        print(f"Total estimated: {memory_estimate['total_estimated_gb']} GB")
        
        # Basic assertions
        self.assertGreater(memory_estimate['total_estimated_gb'], 0)
        self.assertGreater(memory_estimate['total_params_millions'], 0)
        self.assertIsInstance(memory_estimate['model_weights_gb'], float)
    
    def test_mixed_precision_memory_savings(self):
        """Test memory savings with mixed precision (float16)."""
        # Same parameters but with float16
        model_positions = 512
        model_embd = 384
        model_layer = 6
        model_head = 6
        batch_size = 24
        seq_len = 512
        
        # Float32 (4 bytes)
        memory_fp32 = estimate_gpu_memory_requirements(
            model_positions=model_positions,
            model_embd=model_embd,
            model_layer=model_layer,
            model_head=model_head,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype_bytes=4
        )
        
        # Float16 (2 bytes)
        memory_fp16 = estimate_gpu_memory_requirements(
            model_positions=model_positions,
            model_embd=model_embd,
            model_layer=model_layer,
            model_head=model_head,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype_bytes=2
        )
        
        print(f"\nMixed Precision Memory Comparison:")
        print(f"Float32 total: {memory_fp32['total_estimated_gb']} GB")
        print(f"Float16 total: {memory_fp16['total_estimated_gb']} GB")
        print(f"Memory savings: {memory_fp32['total_estimated_gb'] - memory_fp16['total_estimated_gb']:.2f} GB")
        
        # Float16 should use less memory
        self.assertLess(memory_fp16['total_estimated_gb'], memory_fp32['total_estimated_gb'])
    
    def test_batch_size_scaling(self):
        """Test how memory scales with batch size."""
        base_params = {
            'model_positions': 512,
            'model_embd': 384,
            'model_layer': 6,
            'model_head': 6,
            'seq_len': 512
        }
        
        batch_sizes = [1, 4, 8, 16, 24, 32]
        memory_usage = []
        
        for batch_size in batch_sizes:
            memory = estimate_gpu_memory_requirements(
                batch_size=batch_size,
                **base_params
            )
            memory_usage.append((batch_size, memory['total_estimated_gb']))
        
        print(f"\nBatch Size vs Memory Usage:")
        for batch_size, memory_gb in memory_usage:
            print(f"  Batch size {batch_size}: {memory_gb:.2f} GB")
        
        # Memory should increase with batch size
        for i in range(1, len(memory_usage)):
            self.assertGreater(memory_usage[i][1], memory_usage[i-1][1])


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
