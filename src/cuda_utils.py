#!/usr/bin/env python3
"""
CUDA Utilities for AGI2

This module provides essential CUDA availability checks that can be imported
at the start of processing scripts to ensure proper GPU support.
"""

import platform
import subprocess
from pathlib import Path
import warnings


def quick_cuda_check():
    """
    Perform a quick CUDA availability check.
    
    Returns:
        bool: True if CUDA is available and working, False otherwise
    """
    try:
        import torch
        if torch.cuda.is_available():
            return True
        else:
            return False
    except ImportError:
        return False


def check_cuda_availability(verbose=True):
    """
    Check CUDA availability and provide detailed information.
    
    Args:
        verbose (bool): Whether to print detailed information
        
    Returns:
        dict: Dictionary containing CUDA status information
    """
    status = {
        'cuda_available': False,
        'pytorch_cuda': False,
        'nvidia_driver': False,
        'cuda_toolkit': False,
        'gpu_count': 0,
        'device_info': []
    }
    
    if verbose:
        print("=" * 60)
        print("CUDA AVAILABILITY CHECK")
        print("=" * 60)
    
    # Check PyTorch CUDA support
    try:
        import torch
        if verbose:
            print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            status['pytorch_cuda'] = True
            status['cuda_available'] = True
            status['gpu_count'] = torch.cuda.device_count()
            
            if verbose:
                print("✅ CUDA is available in PyTorch")
                print(f"  CUDA version: {torch.version.cuda}")
                print(f"  cuDNN version: {torch.backends.cudnn.version()}")
                print(f"  Number of GPUs: {torch.cuda.device_count()}")
            
            # Get GPU information
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                device_info = {
                    'index': i,
                    'name': gpu_name,
                    'memory_gb': gpu_memory
                }
                status['device_info'].append(device_info)
                
                if verbose:
                    print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            if verbose:
                print("❌ CUDA is NOT available in PyTorch")
                
    except ImportError:
        if verbose:
            print("❌ PyTorch is not installed")
        return status
    
    # Check NVIDIA driver
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            status['nvidia_driver'] = True
            if verbose:
                print("✅ NVIDIA driver is working")
        else:
            if verbose:
                print("❌ NVIDIA driver is not working properly")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        if verbose:
            print("❌ NVIDIA driver not accessible")
    
    # Check CUDA toolkit
    cuda_paths = [
        "/usr/local/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
        "/opt/cuda"
    ]
    
    for cuda_path in cuda_paths:
        if Path(cuda_path).exists():
            status['cuda_toolkit'] = True
            if verbose:
                print(f"✅ CUDA toolkit found at: {cuda_path}")
            break
    
    if not status['cuda_toolkit'] and verbose:
        print("❌ CUDA toolkit not found in common locations")
    
    if verbose:
        print("=" * 60)
        
        if status['cuda_available']:
            print("🎉 Your system is ready for GPU-accelerated processing!")
        else:
            print("⚠️  CUDA is not available - processing will use CPU")
        print("=" * 60)
    
    return status


def ensure_cuda_available(require_cuda=False, verbose=True):
    """
    Ensure CUDA is available, optionally requiring it.
    
    Args:
        require_cuda (bool): If True, exit if CUDA is not available
        verbose (bool): Whether to print detailed information
        
    Returns:
        bool: True if CUDA is available, False otherwise
    """
    status = check_cuda_availability(verbose=verbose)
    
    if require_cuda and not status['cuda_available']:
        print("❌ CUDA is required but not available!")
        print("   Please ensure CUDA is properly installed and PyTorch has CUDA support")
        return False
    
    return status['cuda_available']


def get_optimal_device(device_preference="auto"):
    """
    Get the optimal device for processing.
    
    Args:
        device_preference (str): Device preference ("cpu", "cuda", or "auto")
        
    Returns:
        str: Device to use ("cpu" or "cuda")
    """
    if device_preference == "auto":
        return "cuda" if quick_cuda_check() else "cpu"
    elif device_preference == "cuda":
        if quick_cuda_check():
            return "cuda"
        else:
            warnings.warn("CUDA requested but not available, falling back to CPU")
            return "cpu"
    else:
        return device_preference


def test_gpu_operations():
    """
    Test basic GPU operations to ensure everything is working.
    
    Returns:
        bool: True if GPU operations succeed, False otherwise
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return False
        
        # Create a simple tensor and move it to GPU
        x = torch.randn(100, 100)
        x_gpu = x.cuda()
        
        # Test GPU computation
        result = torch.mm(x_gpu, x_gpu)
        result_cpu = result.cpu()
        
        return True
        
    except Exception:
        return False
