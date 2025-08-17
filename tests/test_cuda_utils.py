#!/usr/bin/env python3
"""
Unit tests for CUDA utilities module.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cuda_utils import (
    quick_cuda_check,
    check_cuda_availability,
    ensure_cuda_available,
    get_optimal_device,
    test_gpu_operations
)


class TestCudaUtils(unittest.TestCase):
    """Test cases for CUDA utilities functions."""
    
    @patch('builtins.__import__')
    def test_quick_cuda_check_with_cuda(self, mock_import):
        """Test quick_cuda_check when CUDA is available."""
        # Mock torch import
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_import.return_value = mock_torch
        
        result = quick_cuda_check()
        self.assertTrue(result)
        mock_torch.cuda.is_available.assert_called_once()
    
    @patch('builtins.__import__')
    def test_quick_cuda_check_without_cuda(self, mock_import):
        """Test quick_cuda_check when CUDA is not available."""
        # Mock torch import
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_import.return_value = mock_torch
        
        result = quick_cuda_check()
        self.assertFalse(result)
    
    @patch('builtins.__import__')
    def test_quick_cuda_check_import_error(self, mock_import):
        """Test quick_cuda_check when torch import fails."""
        mock_import.side_effect = ImportError("torch not found")
        result = quick_cuda_check()
        self.assertFalse(result)
    
    @patch('builtins.__import__')
    @patch('cuda_utils.subprocess.run')
    @patch('cuda_utils.Path')
    def test_check_cuda_availability_with_cuda(self, mock_path, mock_subprocess, mock_import):
        """Test check_cuda_availability when CUDA is fully available."""
        # Mock PyTorch CUDA support
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.0.0"
        mock_torch.cuda.is_available.return_value = True
        mock_torch.version.cuda = "11.8"
        mock_torch.backends.cudnn.version.return_value = "8.5"
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.get_device_name.side_effect = ["RTX 4090", "RTX 4080"]
        mock_torch.cuda.get_device_properties.return_value.total_memory = 24 * 1024**3  # 24 GB
        
        # Mock torch import
        def mock_import_side_effect(name, *args, **kwargs):
            if name == 'torch':
                return mock_torch
            return MagicMock()
        mock_import.side_effect = mock_import_side_effect
        
        # Mock NVIDIA driver check
        mock_subprocess.return_value.returncode = 0
        
        # Mock CUDA toolkit path
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        result = check_cuda_availability(verbose=False)
        
        self.assertTrue(result['cuda_available'])
        self.assertTrue(result['pytorch_cuda'])
        self.assertTrue(result['nvidia_driver'])
        self.assertTrue(result['cuda_toolkit'])
        self.assertEqual(result['gpu_count'], 2)
        self.assertEqual(len(result['device_info']), 2)
    
    @patch('builtins.__import__')
    @patch('cuda_utils.subprocess.run')
    @patch('cuda_utils.Path')
    def test_check_cuda_availability_without_cuda(self, mock_path, mock_subprocess, mock_import):
        """Test check_cuda_availability when CUDA is not available."""
        # Mock PyTorch without CUDA
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.0.0"
        mock_torch.cuda.is_available.return_value = False
        
        # Mock torch import
        def mock_import_side_effect(name, *args, **kwargs):
            if name == 'torch':
                return mock_torch
            return MagicMock()
        mock_import.side_effect = mock_import_side_effect
        
        # Mock subprocess failure for nvidia-smi
        mock_subprocess.side_effect = FileNotFoundError("nvidia-smi not found")
        
        # Mock CUDA toolkit path not found
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance
        
        result = check_cuda_availability(verbose=False)
        
        self.assertFalse(result['cuda_available'])
        self.assertFalse(result['pytorch_cuda'])
        self.assertFalse(result['nvidia_driver'])
        self.assertFalse(result['cuda_toolkit'])
        self.assertEqual(result['gpu_count'], 0)
    
    def test_get_optimal_device_auto_with_cuda(self):
        """Test get_optimal_device with auto preference when CUDA is available."""
        with patch('cuda_utils.quick_cuda_check', return_value=True):
            device = get_optimal_device("auto")
            self.assertEqual(device, "cuda")
    
    def test_get_optimal_device_auto_without_cuda(self):
        """Test get_optimal_device with auto preference when CUDA is not available."""
        with patch('cuda_utils.quick_cuda_check', return_value=False):
            device = get_optimal_device("auto")
            self.assertEqual(device, "cpu")
    
    def test_get_optimal_device_cuda_available(self):
        """Test get_optimal_device with cuda preference when CUDA is available."""
        with patch('cuda_utils.quick_cuda_check', return_value=True):
            device = get_optimal_device("cuda")
            self.assertEqual(device, "cuda")
    
    def test_get_optimal_device_cuda_unavailable(self):
        """Test get_optimal_device with cuda preference when CUDA is not available."""
        with patch('cuda_utils.quick_cuda_check', return_value=False):
            with patch('cuda_utils.warnings.warn') as mock_warn:
                device = get_optimal_device("cuda")
                self.assertEqual(device, "cpu")
                mock_warn.assert_called_once()
    
    def test_get_optimal_device_cpu(self):
        """Test get_optimal_device with cpu preference."""
        device = get_optimal_device("cpu")
        self.assertEqual(device, "cpu")
    
    @patch('builtins.__import__')
    def test_test_gpu_operations_success(self, mock_import):
        """Test test_gpu_operations when GPU operations succeed."""
        # Mock torch import
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.randn.return_value = MagicMock()
        mock_torch.mm.return_value = MagicMock()
        
        def mock_import_side_effect(name, *args, **kwargs):
            if name == 'torch':
                return mock_torch
            return MagicMock()
        mock_import.side_effect = mock_import_side_effect
        
        result = test_gpu_operations()
        self.assertTrue(result)
    
    @patch('builtins.__import__')
    def test_test_gpu_operations_no_cuda(self, mock_import):
        """Test test_gpu_operations when CUDA is not available."""
        # Mock torch import
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        
        def mock_import_side_effect(name, *args, **kwargs):
            if name == 'torch':
                return mock_torch
            return MagicMock()
        mock_import.side_effect = mock_import_side_effect
        
        result = test_gpu_operations()
        self.assertFalse(result)
    
    @patch('builtins.__import__')
    def test_test_gpu_operations_exception(self, mock_import):
        """Test test_gpu_operations when an exception occurs."""
        # Mock torch import
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.randn.side_effect = Exception("GPU error")
        
        def mock_import_side_effect(name, *args, **kwargs):
            if name == 'torch':
                return mock_torch
            return MagicMock()
        mock_import.side_effect = mock_import_side_effect
        
        result = test_gpu_operations()
        self.assertFalse(result)
    
    def test_ensure_cuda_available_not_required(self):
        """Test ensure_cuda_available when CUDA is not required."""
        with patch('cuda_utils.check_cuda_availability') as mock_check:
            mock_check.return_value = {'cuda_available': False}
            result = ensure_cuda_available(require_cuda=False, verbose=False)
            self.assertFalse(result)
    
    def test_ensure_cuda_available_required_and_available(self):
        """Test ensure_cuda_available when CUDA is required and available."""
        with patch('cuda_utils.check_cuda_availability') as mock_check:
            mock_check.return_value = {'cuda_available': True}
            result = ensure_cuda_available(require_cuda=True, verbose=False)
            self.assertTrue(result)
    
    def test_ensure_cuda_available_required_but_unavailable(self):
        """Test ensure_cuda_available when CUDA is required but unavailable."""
        with patch('cuda_utils.check_cuda_availability') as mock_check:
            mock_check.return_value = {'cuda_available': False}
            result = ensure_cuda_available(require_cuda=True, verbose=False)
            self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
