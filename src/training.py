"""
Training Functions

This module provides training functions and training loop for the AGI2 model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import os
import time
from typing import Optional, Dict, Any

from .dataset import TextDataset
from .utils import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad_norm: float = 1.0,
    use_amp: bool = True,
    log_gpu_memory: bool = False
) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The AGI2 model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to train on
        clip_grad_norm: Gradient clipping norm value
        use_amp: Whether to use automatic mixed precision
        log_gpu_memory: Whether to log GPU memory usage
        
    Returns:
        Dictionary containing training metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Initialize AMP scaler if using mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    
    # Cache GPU memory info to avoid repeated queries
    gpu_memory_info = ""
    if log_gpu_memory and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        gpu_memory_info = f", GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch = batch.to(device, non_blocking=True)
        
        # Prepare input and target
        input_ids = batch[:, :-1]  # All tokens except last
        target_ids = batch[:, 1:]  # All tokens except first
        
        # Forward pass with AMP if enabled
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_ids)
                loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            
            # Backward pass with AMP
            scaler.scale(loss).backward()
            
            # Gradient clipping with AMP
            if clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            # Update weights with AMP
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training without AMP
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            # Update weights
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}{gpu_memory_info}")
    
    return total_loss / num_batches


def train_model(
    model,
    tokenizer,
    sources: str | list[str],
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    seq_len: int = 512,
    device: str = "auto",
    save_path: str = "model",
    start_epoch: int = 0,
    resume_path: Optional[str] = None,
    use_amp: bool = True,
    log_gpu_memory: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True
) -> None:
    """
    Train the AGI2 model on text data.
    
    Args:
        model: The AGI2 model to train
        tokenizer: Tokenizer for text processing
        sources: Path to training corpus file(s) - can be single path or list of paths
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        seq_len: Maximum sequence length for training
        device: Device to train on (auto, cpu, cuda)
        save_path: Base path for saving checkpoints
        start_epoch: Starting epoch (for resume training)
        resume_path: Path to resume checkpoint (optional)
        use_amp: Whether to use automatic mixed precision
        log_gpu_memory: Whether to log GPU memory usage
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
    """
    device = torch.device(device)
    model = model.to(device)
    
    # Create dataset and dataloader with optimizations
    dataset = TextDataset(sources, tokenizer, seq_len)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and device.type == 'cuda',
        persistent_workers=num_workers > 0
    )
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'epoch_times': []
    }
    
    # If resuming, load optimizer state from checkpoint
    if start_epoch > 0:
        print(f"Resuming training from epoch {start_epoch + 1}")
        print(f"Note: Optimizer state will be reinitialized for simplicity")
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Dataset size: {len(dataset)} sequences")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")
    print(f"DataLoader workers: {num_workers}")
    print(f"Pin memory: {pin_memory}")
    
    # GPU monitoring commands for reference
    if torch.cuda.is_available():
        print("\nGPU Memory Monitoring Commands:")
        print("  nvidia-smi -l 1")
        print("  nvidia-smi -l 1 --query-gpu=memory.used,memory.total,memory.free --format=csv")
        
        # Estimate GPU memory requirements
        try:
            from .utils import estimate_gpu_memory_requirements
            
            # Get model configuration from the model
            config = model.config
            memory_estimate = estimate_gpu_memory_requirements(
                model_positions=config.get('max_positions', 512),
                model_embd=config.get('n_embd', 384),
                model_layer=config.get('n_layer', 6),
                model_head=config.get('n_head', 6),
                batch_size=batch_size,
                seq_len=seq_len,
                dtype_bytes=2 if use_amp else 4  # 2 for mixed precision, 4 for float32
            )
            
            print(f"\nGPU Memory Requirements Estimate:")
            print(f"  Model parameters: {memory_estimate['total_params_millions']}M")
            print(f"  Estimated GPU memory: {memory_estimate['total_estimated_gb']:.2f} GB")
            print(f"  Breakdown:")
            print(f"    - Model weights: {memory_estimate['model_weights_gb']:.2f} GB")
            print(f"    - Activations: {memory_estimate['activations_gb']:.2f} GB")
            print(f"    - Optimizer state: {memory_estimate['optimizer_state_gb']:.2f} GB")
            print(f"    - Gradients: {memory_estimate['gradients_gb']:.2f} GB")
            print(f"    - IO tensors: {memory_estimate['io_tensors_gb']:.2f} GB")
            
            # Check if this might exceed common GPU memory limits
            if memory_estimate['total_estimated_gb'] > 8:
                print(f"  ⚠️  High memory usage - consider reducing batch_size or seq_len")
            elif memory_estimate['total_estimated_gb'] > 4:
                print(f"  ⚠️  Moderate memory usage - monitor GPU memory during training")
            else:
                print(f"  ✅ Memory usage looks reasonable")
                
        except ImportError:
            print("  (Memory estimation not available)")
        except Exception as e:
            print(f"  (Memory estimation failed: {e})")
    
    # Create trained directory if it doesn't exist
    if save_path:
        import os
        os.makedirs("trained", exist_ok=True)
    
    for epoch in range(start_epoch, start_epoch + epochs):
        start_time = time.time()
        
        print(f"\nEpoch {epoch + 1}/{start_epoch + epochs}")
        print("-" * 50)
        
        # Show GPU memory before training (only if logging is enabled)
        if log_gpu_memory and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory before training: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # Train for one epoch with optimizations
        avg_loss = train_epoch(
            model, dataloader, optimizer, criterion, device, 
            use_amp=use_amp, log_gpu_memory=log_gpu_memory
        )
        
        epoch_time = time.time() - start_time
        
        # Show GPU memory after training (only if logging is enabled)
        if log_gpu_memory and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory after training: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # Record history
        history['train_loss'].append(avg_loss)
        history['epoch_times'].append(epoch_time)
        
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if save_path and (epoch + 1) % 5 == 0:
            # Extract just the filename from the save_path to avoid path issues
            model_name = Path(save_path).stem if '/' in str(save_path) or '\\' in str(save_path) else save_path
            checkpoint_path = f"trained/{model_name}.pt_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': model.config,
                'tokenizer': tokenizer
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    if save_path:
        # Extract just the filename from the save_path to avoid path issues
        model_name = Path(save_path).stem if '/' in str(save_path) or '\\' in str(save_path) else save_path
        final_path = f"trained/{model_name}.pt"
        torch.save({
            'epoch': start_epoch + epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': history['train_loss'][-1],
            'config': model.config,
            'tokenizer': tokenizer
        }, final_path)
        print(f"Final model saved: {final_path}")
    
    print("\nTraining completed!")
    print(f"Final loss: {history['train_loss'][-1]:.4f}")
    
    return history
