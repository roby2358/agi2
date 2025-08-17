#!/usr/bin/env python3
"""
CUDA Availability Checker

This utility checks whether CUDA is available on your system and provides
detailed information about GPU support for PyTorch.
"""

import sys
import platform
import subprocess
from pathlib import Path


def check_system_info():
    """Display basic system information."""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print()


def check_nvidia_driver():
    """Check if NVIDIA driver is installed and working."""
    print("=" * 60)
    print("NVIDIA DRIVER CHECK")
    print("=" * 60)
    
    try:
        # Try to run nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA driver is installed and working")
            print("GPU Information:")
            # Extract GPU info from nvidia-smi output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA' in line and 'GeForce' in line:
                    print(f"  {line.strip()}")
                elif 'Driver Version' in line:
                    print(f"  {line.strip()}")
        else:
            print("❌ NVIDIA driver is not working properly")
            print(f"Error: {result.stderr}")
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA driver may not be installed")
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi timed out - driver may be unresponsive")
    except Exception as e:
        print(f"❌ Error checking NVIDIA driver: {e}")
    print()


def check_cuda_toolkit():
    """Check if CUDA toolkit is installed."""
    print("=" * 60)
    print("CUDA TOOLKIT CHECK")
    print("=" * 60)
    
    # Check common CUDA installation paths
    cuda_paths = [
        "/usr/local/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
        "/opt/cuda"
    ]
    
    cuda_found = False
    for cuda_path in cuda_paths:
        if Path(cuda_path).exists():
            print(f"✅ CUDA toolkit found at: {cuda_path}")
            cuda_found = True
            
            # Try to get CUDA version
            nvcc_path = Path(cuda_path) / "bin" / "nvcc"
            if nvcc_path.exists():
                try:
                    result = subprocess.run([str(nvcc_path), '--version'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        version_line = result.stdout.split('\n')[0]
                        print(f"  CUDA Version: {version_line}")
                except Exception:
                    pass
            break
    
    if not cuda_found:
        print("❌ CUDA toolkit not found in common locations")
        print("   You may need to install CUDA toolkit from NVIDIA")
    
    print()


def check_pytorch_cuda():
    """Check PyTorch CUDA support."""
    print("=" * 60)
    print("PYTORCH CUDA SUPPORT")
    print("=" * 60)
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print("✅ CUDA is available in PyTorch")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            
            # Display GPU information
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
            # Check current device
            if torch.cuda.current_device() >= 0:
                current_device = torch.cuda.current_device()
                print(f"  Current GPU device: {current_device}")
                
        else:
            print("❌ CUDA is NOT available in PyTorch")
            print("   This could mean:")
            print("   - PyTorch was installed without CUDA support")
            print("   - CUDA toolkit is not compatible with PyTorch version")
            print("   - GPU drivers are not properly installed")
            
        # Check if PyTorch was built with CUDA
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            print(f"  PyTorch was built with CUDA: {torch.version.cuda}")
        else:
            print("  PyTorch was built WITHOUT CUDA support")
            
    except ImportError:
        print("❌ PyTorch is not installed")
        print("   Install PyTorch with: pip install torch")
    except Exception as e:
        print(f"❌ Error checking PyTorch: {e}")
    
    print()


def check_tensor_operations():
    """Test basic tensor operations on GPU if available."""
    print("=" * 60)
    print("GPU TENSOR OPERATIONS TEST")
    print("=" * 60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            # Create a simple tensor and move it to GPU
            x = torch.randn(1000, 1000)
            print(f"✅ Created tensor on CPU: {x.device}")
            
            # Move to GPU
            x_gpu = x.cuda()
            print(f"✅ Moved tensor to GPU: {x_gpu.device}")
            
            # Test GPU computation
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            result = torch.mm(x_gpu, x_gpu)
            end_time.record()
            
            torch.cuda.synchronize()
            gpu_time = start_time.elapsed_time(end_time)
            
            print(f"✅ GPU matrix multiplication successful")
            print(f"  Computation time: {gpu_time:.2f} ms")
            print(f"  Result shape: {result.shape}")
            
            # Move back to CPU
            result_cpu = result.cpu()
            print(f"✅ Moved result back to CPU: {result_cpu.device}")
            
        else:
            print("❌ CUDA not available - cannot test GPU operations")
            
    except Exception as e:
        print(f"❌ Error testing GPU operations: {e}")
    
    print()


def main():
    """Main function to run all checks."""
    print("🚀 CUDA AVAILABILITY CHECKER")
    print("This utility checks your system for CUDA support and PyTorch GPU capabilities")
    print()
    
    check_system_info()
    check_nvidia_driver()
    check_cuda_toolkit()
    check_pytorch_cuda()
    check_tensor_operations()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print("🎉 Your system is ready for GPU-accelerated PyTorch training!")
            print("   You can use device='cuda' in your training scripts")
        else:
            print("⚠️  CUDA is not available in PyTorch")
            print("   Consider reinstalling PyTorch with CUDA support:")
            print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
    except ImportError:
        print("❌ PyTorch is not installed")
        print("   Install PyTorch first: pip install torch")
    
    print()
    print("For more information, visit: https://pytorch.org/get-started/locally/")


if __name__ == "__main__":
    main()
