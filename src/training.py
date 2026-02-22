"""
Training Functions

Training loop for the AGI2 model using pairwise cosine similarity loss.
Only the last hidden vector is compared — we don't care how it got there,
just where it landed.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .cosine_loss import PairwiseCosineLoss
from .dataset import TextDataset

logger = logging.getLogger(__name__)


def _collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Pad variable-length sequences in a batch to uniform length."""
    prompt_ids = [item["prompt_ids"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]

    max_prompt_len = max(p.size(0) for p in prompt_ids)
    padded_prompts = torch.zeros(len(batch), max_prompt_len, dtype=torch.long)
    prompt_masks = torch.zeros(len(batch), max_prompt_len, dtype=torch.bool)
    for i, p in enumerate(prompt_ids):
        padded_prompts[i, : p.size(0)] = p
        prompt_masks[i, : p.size(0)] = True

    max_target_len = max(t.size(0) for t in target_ids)
    padded_targets = torch.zeros(len(batch), max_target_len, dtype=torch.long)
    for i, t in enumerate(target_ids):
        padded_targets[i, : t.size(0)] = t

    return {
        "prompt_ids": padded_prompts,
        "prompt_mask": prompt_masks,
        "target_ids": padded_targets,
    }


def _compute_batch_loss(
    model: nn.Module,
    full_input: torch.Tensor,
    prompt_ids: torch.Tensor,
    target_ids: torch.Tensor,
    loss_fn: PairwiseCosineLoss,
) -> tuple:
    """Compute pairwise cosine similarity loss for a single batch.

    Uses only the last hidden vector — the final position of the target
    sequence after running through the transformer.
    """
    _, hidden_states = model.forward_hidden(full_input)

    # Last hidden vector of the target region
    prompt_len = prompt_ids.size(1)
    target_len = target_ids.size(1)
    last_target_pos = prompt_len + target_len - 1
    h = hidden_states[:, last_target_pos, :]

    # Last target embedding from the frozen codebook
    embedding_weight = model.token_embeddings.embedding.weight
    last_target_token = target_ids[:, -1]
    e = embedding_weight[last_target_token]

    return loss_fn(h, e, embedding_weight)


def _step_with_amp(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    full_input: torch.Tensor,
    prompt_ids: torch.Tensor,
    target_ids: torch.Tensor,
    loss_fn: PairwiseCosineLoss,
    clip_grad_norm: float,
) -> tuple:
    """Forward + backward with AMP."""
    with torch.cuda.amp.autocast():
        loss, metrics = _compute_batch_loss(
            model,
            full_input,
            prompt_ids,
            target_ids,
            loss_fn,
        )
    scaler.scale(loss).backward()
    if clip_grad_norm > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    return loss, metrics


def _step_standard(
    model: nn.Module,
    optimizer: optim.Optimizer,
    full_input: torch.Tensor,
    prompt_ids: torch.Tensor,
    target_ids: torch.Tensor,
    loss_fn: PairwiseCosineLoss,
    clip_grad_norm: float,
) -> tuple:
    """Forward + backward without AMP."""
    loss, metrics = _compute_batch_loss(
        model,
        full_input,
        prompt_ids,
        target_ids,
        loss_fn,
    )
    loss.backward()
    if clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
    optimizer.step()
    return loss, metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: PairwiseCosineLoss,
    device: torch.device,
    clip_grad_norm: float,
    scaler: torch.cuda.amp.GradScaler | None,
    log_gpu_memory: bool,
) -> Dict[str, float]:
    """
    Train the model for one epoch using pairwise cosine similarity loss.

    Returns dictionary of averaged training metrics for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_metrics: Dict[str, float] = {}
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        prompt_ids = batch["prompt_ids"].to(device, non_blocking=True)
        target_ids = batch["target_ids"].to(device, non_blocking=True)
        full_input = torch.cat([prompt_ids, target_ids], dim=1)

        optimizer.zero_grad()

        if scaler is not None:
            loss, metrics = _step_with_amp(
                model,
                optimizer,
                scaler,
                full_input,
                prompt_ids,
                target_ids,
                loss_fn,
                clip_grad_norm,
            )
        else:
            loss, metrics = _step_standard(
                model,
                optimizer,
                full_input,
                prompt_ids,
                target_ids,
                loss_fn,
                clip_grad_norm,
            )

        total_loss += loss.item()
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        num_batches += 1

        if (batch_idx + 1) % 100 == 0:
            gpu_info = ""
            if log_gpu_memory and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                gpu_info = f", GPU: {allocated:.2f}GB alloc, {reserved:.2f}GB res"
            print(
                f"Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.4f}{gpu_info}"
            )

    avg_loss = total_loss / max(num_batches, 1)
    avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}
    avg_metrics["avg_loss"] = avg_loss
    return avg_metrics


def _build_dataloader(
    dataset: TextDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    is_cuda: bool,
) -> DataLoader:
    """Build a DataLoader with the correct settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and is_cuda,
        persistent_workers=num_workers > 0,
        collate_fn=_collate_fn,
    )


def train_model(
    model: nn.Module,
    tokenizer: object,
    sources: str | list[str],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seq_len_start: int,
    seq_len_end: int,
    device: str,
    save_path: str,
    start_epoch: int,
    use_amp: bool,
    log_gpu_memory: bool,
    num_workers: int,
    pin_memory: bool,
    geometric_ratio: float,
    anchor_ratio: float,
    sigmoid_scale_start: float,
    sigmoid_scale_end: float,
    early_stop_patience: int,
) -> Dict[str, Any]:
    """
    Train the AGI2 model using pairwise cosine similarity loss.

    Sigmoid scale ramps linearly from sigmoid_scale_start to sigmoid_scale_end
    over the training run, gradually tightening tolerances as the model improves.

    Returns training history dict with keys:
    train_loss, epoch_times, metrics.
    """
    device_obj = torch.device(device)
    model = model.to(device_obj)
    is_cuda = device_obj.type == "cuda"

    dataset = TextDataset(sources, tokenizer, seq_len_start)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = PairwiseCosineLoss(geometric_ratio, anchor_ratio, sigmoid_scale_start)
    scaler = (
        torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    )

    history: Dict[str, Any] = {
        "train_loss": [],
        "epoch_times": [],
        "metrics": [],
    }

    best_loss = float("inf")
    patience_counter = 0
    prev_seq_len = seq_len_start

    if start_epoch > 0:
        print(f"Resuming training from epoch {start_epoch + 1}")

    print(f"Starting training for {epochs} epochs...")
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Batch size: {batch_size}, LR: {learning_rate}")
    print(f"Loss ratios: geometric={geometric_ratio}, anchor={anchor_ratio}")
    print(f"Seq len: {seq_len_start} -> {seq_len_end} over {epochs} epochs")
    print(
        f"Sigmoid scale: {sigmoid_scale_start} -> {sigmoid_scale_end} over {epochs} epochs"
    )
    print(f"Mixed Precision: {'Enabled' if scaler is not None else 'Disabled'}")

    if save_path:
        os.makedirs("trained", exist_ok=True)

    for epoch in range(start_epoch, start_epoch + epochs):
        # Ramp sigmoid scale and seq_len linearly
        total_epochs = start_epoch + epochs
        progress = epoch / max(total_epochs - 1, 1)
        current_scale = (
            sigmoid_scale_start + (sigmoid_scale_end - sigmoid_scale_start) * progress
        )
        loss_fn.sigmoid_scale = current_scale

        current_seq_len = int(seq_len_start + (seq_len_end - seq_len_start) * progress)

        # Rebuild dataloader when seq_len changes
        if current_seq_len != prev_seq_len or epoch == start_epoch:
            dataset.set_seq_len(current_seq_len)
            prev_seq_len = current_seq_len

        dataloader = _build_dataloader(
            dataset,
            batch_size,
            num_workers,
            pin_memory,
            is_cuda,
        )

        start_time = time.time()
        print(
            f"\nEpoch {epoch + 1}/{total_epochs} "
            f"(seq={current_seq_len}, scale={current_scale:.2f})"
        )
        print("-" * 50)

        epoch_metrics = train_epoch(
            model,
            dataloader,
            optimizer,
            loss_fn,
            device_obj,
            1.0,
            scaler,
            log_gpu_memory,
        )

        epoch_time = time.time() - start_time
        avg_loss = epoch_metrics["avg_loss"]

        history["train_loss"].append(avg_loss)
        history["epoch_times"].append(epoch_time)
        history["metrics"].append(epoch_metrics)

        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"Average loss: {avg_loss:.4f}")
        for k, v in epoch_metrics.items():
            if k != "avg_loss":
                print(f"  {k}: {v:.4f}")

        # Early stop: raw gap collapsed to zero
        raw_gap = epoch_metrics.get("raw_gap", avg_loss)
        if raw_gap < 1e-6:
            print(f"\nEarly stop: raw gap collapsed to {raw_gap:.2e}")
            break

        # Early stop: raw gap plateaued (scale-independent)
        if raw_gap < best_loss:
            best_loss = raw_gap
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"\nEarly stop: loss plateaued for {early_stop_patience} epochs")
            break

        # Checkpoint every 5 epochs
        if save_path and (epoch + 1) % 5 == 0:
            _save_checkpoint(
                model,
                optimizer,
                tokenizer,
                avg_loss,
                epoch + 1,
                save_path,
                is_final=False,
            )

    # Save final model
    if save_path:
        _save_checkpoint(
            model,
            optimizer,
            tokenizer,
            history["train_loss"][-1],
            start_epoch + epochs,
            save_path,
            is_final=True,
        )

    print("\nTraining completed!")
    print(f"Final loss: {history['train_loss'][-1]:.4f}")
    return history


def _save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    tokenizer: object,
    loss: float,
    epoch: int,
    save_path: str,
    is_final: bool,
) -> None:
    """Save a training checkpoint."""
    model_name = (
        Path(save_path).stem
        if "/" in str(save_path) or "\\" in str(save_path)
        else save_path
    )
    if is_final:
        path = f"trained/{model_name}.pt"
    else:
        path = f"trained/{model_name}.pt_epoch_{epoch}.pt"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "config": model.config,
            "tokenizer": tokenizer,
        },
        path,
    )
    label = "Final model" if is_final else "Checkpoint"
    print(f"{label} saved: {path}")
