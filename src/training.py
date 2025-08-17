"""
Training Functions

This module provides training functions and training loop for the GPT-2 model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import time


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad_norm: float = 1.0
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: The GPT-2 model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for updating weights
        criterion: Loss function
        device: Device to train on
        clip_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch = batch.to(device)
        
        # Prepare input and target
        input_ids = batch[:, :-1]  # All tokens except last
        target_ids = batch[:, 1:]  # All tokens except first
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids)
        
        # Calculate loss
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
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
            print(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def train_model(
    model: nn.Module,
    tokenizer,
    corpus_path: str,
    epochs: int,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    seq_len: int = 1024,
    device: str = "cpu",
    save_path: Optional[str] = None,
    **kwargs
) -> Dict[str, List[float]]:
    """
    Train the model from start to finish.
    
    Args:
        model: The GPT-2 model to train
        tokenizer: Pre-fitted tokenizer to use for text processing
        corpus_path: Path to the text corpus file
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        seq_len: Sequence length for training
        device: Device to train on ('cpu' or 'cuda')
        save_path: Path to save the trained model
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing training history
    """
    device = torch.device(device)
    model = model.to(device)
    
    # Import dataset class
    from .dataset import TextDataset
    
    # Create dataset and dataloader
    dataset = TextDataset(corpus_path, tokenizer, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'epoch_times': []
    }
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Dataset size: {len(dataset)} sequences")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        # Train for one epoch
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # Record history
        history['train_loss'].append(avg_loss)
        history['epoch_times'].append(epoch_time)
        
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if save_path and (epoch + 1) % 5 == 0:
            checkpoint_path = f"{save_path}_epoch_{epoch + 1}.pt"
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
        final_path = f"{save_path}_final.pt"
        torch.save({
            'epoch': epochs,
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
