"""
Training Functions

Training loop for the AGI2 model using pairwise cosine similarity loss.
Only the hidden vector at the last prompt position is compared — it must
land near the embedding of the next (unseen) target token.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .cosine_loss import CrossEntropyLoss, InfoNCELoss, PairwiseCosineLoss
from .dataset import TextDataset

logger = logging.getLogger(__name__)


def _current_seq_len(
    epoch: int,
    seq_len_start: int,
    seq_len_end: int,
    total_epochs: int,
    ramp_epochs: int,
) -> int:
    """Sequence length for an epoch under the linear ramp.

    ramp_epochs <= 0 spreads the ramp over the whole run; N > 0 completes
    it by epoch N (0-indexed epochs, so epoch N-1 is the first full-length
    epoch) and holds at seq_len_end after. seq_len_start == seq_len_end
    trains at a constant length regardless.
    """
    span = ramp_epochs if ramp_epochs > 0 else total_epochs
    progress = min(1.0, epoch / max(span - 1, 1))
    return int(seq_len_start + (seq_len_end - seq_len_start) * progress)


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


def _next_token_ids(
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-position next-token ids for a padded batch.

    Position t's next token is prompt_ids[t + 1] inside the window; the last
    REAL position's (per prompt_mask) is the window's held-out target. Values
    at padding positions are meaningless — callers must exclude them with
    prompt_mask before use.
    """
    next_ids = torch.zeros_like(prompt_ids)
    next_ids[:, :-1] = prompt_ids[:, 1:]
    last_prompt_pos = prompt_mask.sum(dim=1) - 1
    batch_index = torch.arange(prompt_ids.size(0), device=prompt_ids.device)
    next_ids[batch_index, last_prompt_pos] = target_ids[:, 0]
    return next_ids


def _compute_batch_loss(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    target_ids: torch.Tensor,
    loss_fn: PairwiseCosineLoss,
    dense_targets: bool = True,
) -> tuple:
    """Compute pairwise cosine similarity loss for a single batch.

    With dense_targets (default), EVERY real position's hidden vector must
    predict its next token — one training signal per token, exactly as the
    causal model computes hidden states anyway. All valid (hidden, target)
    pairs are flattened into one big observation set, and PairwiseCosineLoss
    already scales its pair sampling with N.

    With dense_targets=False (the control), only the hidden vector at each
    sample's last real prompt position trains against the held-out target
    token. Both formulations match generation, which scores the next token
    from the hidden state at the end of the sequence so far.

    InfoNCELoss and CrossEntropyLoss receive the target token IDS (their
    cross-entropy needs the class index); PairwiseCosineLoss receives the
    target embeddings.
    """
    _, hidden_states = model.forward_hidden(prompt_ids)
    embedding_weight = model.token_embeddings.embedding.weight

    if dense_targets:
        next_ids = _next_token_ids(prompt_ids, prompt_mask, target_ids)
        h = hidden_states[prompt_mask]  # (N, n_embd) — real positions only
        ids = next_ids[prompt_mask]
    else:
        # Hidden vector at the last unpadded prompt position, per sample
        last_prompt_pos = prompt_mask.sum(dim=1) - 1
        batch_index = torch.arange(prompt_ids.size(0), device=prompt_ids.device)
        h = hidden_states[batch_index, last_prompt_pos, :]
        ids = target_ids[:, 0]

    id_targets = isinstance(loss_fn, (InfoNCELoss, CrossEntropyLoss))
    target = ids if id_targets else embedding_weight[ids]
    return loss_fn(h, target, embedding_weight)


def _step_with_amp(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    target_ids: torch.Tensor,
    loss_fn: PairwiseCosineLoss,
    clip_grad_norm: float,
    dense_targets: bool = True,
) -> tuple:
    """Forward + backward with AMP."""
    with torch.cuda.amp.autocast():
        loss, metrics = _compute_batch_loss(
            model,
            prompt_ids,
            prompt_mask,
            target_ids,
            loss_fn,
            dense_targets,
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
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    target_ids: torch.Tensor,
    loss_fn: PairwiseCosineLoss,
    clip_grad_norm: float,
    dense_targets: bool = True,
) -> tuple:
    """Forward + backward without AMP."""
    loss, metrics = _compute_batch_loss(
        model,
        prompt_ids,
        prompt_mask,
        target_ids,
        loss_fn,
        dense_targets,
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
    dense_targets: bool = True,
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
        prompt_mask = batch["prompt_mask"].to(device, non_blocking=True)
        target_ids = batch["target_ids"].to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            loss, metrics = _step_with_amp(
                model,
                optimizer,
                scaler,
                prompt_ids,
                prompt_mask,
                target_ids,
                loss_fn,
                clip_grad_norm,
                dense_targets,
            )
        else:
            loss, metrics = _step_standard(
                model,
                optimizer,
                prompt_ids,
                prompt_mask,
                target_ids,
                loss_fn,
                clip_grad_norm,
                dense_targets,
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


def _evaluate(
    model: nn.Module,
    val_batches: List[Dict[str, torch.Tensor]],
    loss_fn: PairwiseCosineLoss,
    device: torch.device,
    dense_targets: bool,
) -> Dict[str, float]:
    """Held-out metrics (val_-prefixed) over pre-collated validation batches."""
    model.eval()
    totals: Dict[str, float] = {}
    num_batches = 0
    with torch.no_grad():
        for batch in val_batches:
            _, metrics = _compute_batch_loss(
                model,
                batch["prompt_ids"].to(device),
                batch["prompt_mask"].to(device),
                batch["target_ids"].to(device),
                loss_fn,
                dense_targets,
            )
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v
            num_batches += 1
    model.train()
    return {f"val_{k}": v / max(num_batches, 1) for k, v in totals.items()}


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
    align_sequences: bool = True,
    dense_targets: bool = True,
    objective: str = "pairwise",
    nce_temperature: float = 0.07,
    seq_len_ramp_epochs: int = 0,
    val_fraction: float = 0.0,
) -> Dict[str, Any]:
    """
    Train the AGI2 model using pairwise cosine similarity loss.

    With dense_targets (default), every real position trains against its
    next token — see _compute_batch_loss; dense_targets=False restores the
    original one-signal-per-window formulation as a control.

    objective selects the loss: "pairwise" (the sigmoid-gap formulation) or
    "infonce" (cross-entropy over cosine logits at nce_temperature, with
    the geometric term as a weighted auxiliary — anchor_ratio is unused).

    Sigmoid scale ramps linearly from sigmoid_scale_start to sigmoid_scale_end
    over the training run, gradually tightening tolerances as the model improves.

    val_fraction > 0 holds out that fraction of the corpus tail from
    training entirely and reports val_-prefixed held-out metrics every
    epoch (evaluated at seq_len_end), saving the best-val weights to
    trained/<model>_best.pt as they improve. Early stopping then watches
    the held-out metric (val_raw_gap) instead of the training raw gap.
    The history dict is also dumped to trained/<model>_history.json each
    epoch.

    seq_len_ramp_epochs controls how fast seq_len ramps from seq_len_start
    to seq_len_end: 0 (default) spreads the ramp over the whole run, N > 0
    completes it by epoch N and trains at seq_len_end thereafter. The
    sigmoid scale always ramps over the whole run regardless.

    With align_sequences (default), training windows start at utterance
    boundaries when the tokenizer defines an atomic <|endoftext|> token;
    tokenizers without one fall back to stride starts unchanged.

    Returns training history dict with keys:
    train_loss, epoch_times, metrics.
    """
    device_obj = torch.device(device)
    model = model.to(device_obj)
    is_cuda = device_obj.type == "cuda"

    boundary_token = None
    if align_sequences:
        vocab = getattr(tokenizer, "vocab", None)
        if isinstance(vocab, dict):
            boundary_token = vocab.get("<|endoftext|>")

    dataset = TextDataset(
        sources, tokenizer, seq_len_start, boundary_token, val_fraction
    )

    # Pre-collate held-out batches once — fixed at seq_len_end so val
    # metrics are comparable across epochs regardless of the seq_len ramp
    val_batches: List[Dict[str, torch.Tensor]] = []
    if val_fraction > 0.0:
        windows = dataset.val_windows(seq_len_end)
        val_batches = [
            _collate_fn(windows[i : i + batch_size])
            for i in range(0, len(windows), batch_size)
        ]
        print(f"Validation: {len(windows)} windows in {len(val_batches)} batches")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    if objective == "infonce":
        loss_fn: PairwiseCosineLoss = InfoNCELoss(
            geometric_ratio, sigmoid_scale_start, nce_temperature
        )
    elif objective == "ce":
        loss_fn = CrossEntropyLoss()
    elif objective == "pairwise":
        loss_fn = PairwiseCosineLoss(geometric_ratio, anchor_ratio, sigmoid_scale_start)
    else:
        raise ValueError(f"Unknown objective: {objective!r} (pairwise | infonce | ce)")
    scaler = torch.cuda.amp.GradScaler() if use_amp and is_cuda else None

    history: Dict[str, Any] = {
        "train_loss": [],
        "epoch_times": [],
        "metrics": [],
    }

    best_loss = float("inf")
    best_val_loss = float("inf")
    patience_counter = 0
    prev_seq_len = seq_len_start

    training_start_time = time.time()

    if start_epoch > 0:
        print(f"Resuming training from epoch {start_epoch + 1}")

    print(f"Starting training for {epochs} epochs...")
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Batch size: {batch_size}, LR: {learning_rate}")
    if objective == "infonce":
        print(
            f"Objective: infonce (temp {nce_temperature}), "
            f"geometric auxiliary ratio {geometric_ratio}"
        )
    elif objective == "ce":
        print("Objective: ce (plain cross-entropy over tied-projection logits)")
    else:
        print(f"Loss ratios: geometric={geometric_ratio}, anchor={anchor_ratio}")
    ramp_desc = (
        f"{seq_len_ramp_epochs}" if seq_len_ramp_epochs > 0 else f"{epochs} (whole run)"
    )
    print(f"Seq len: {seq_len_start} -> {seq_len_end} over {ramp_desc} epochs")
    print(
        f"Sigmoid scale: {sigmoid_scale_start} -> {sigmoid_scale_end} over {epochs} epochs"
    )
    print(f"Mixed Precision: {'Enabled' if scaler is not None else 'Disabled'}")

    if save_path:
        os.makedirs("trained", exist_ok=True)

    for epoch in range(start_epoch, start_epoch + epochs):
        # Ramp sigmoid scale (whole run) and seq_len (seq_len_ramp_epochs)
        total_epochs = start_epoch + epochs
        progress = epoch / max(total_epochs - 1, 1)
        current_scale = (
            sigmoid_scale_start + (sigmoid_scale_end - sigmoid_scale_start) * progress
        )
        loss_fn.sigmoid_scale = current_scale

        current_seq_len = _current_seq_len(
            epoch, seq_len_start, seq_len_end, total_epochs, seq_len_ramp_epochs
        )

        # Scale batch size inversely with seq_len to keep memory constant.
        # 1.125x headroom lets shorter sequences use more memory.
        token_budget = int(batch_size * seq_len_end * 1.125)
        current_batch_size = max(4, token_budget // max(current_seq_len, 1))

        # Rebuild dataloader when seq_len changes
        if current_seq_len != prev_seq_len or epoch == start_epoch:
            dataset.set_seq_len(current_seq_len)
            prev_seq_len = current_seq_len

        dataloader = _build_dataloader(
            dataset,
            current_batch_size,
            num_workers,
            pin_memory,
            is_cuda,
        )

        start_time = time.time()
        print(
            f"\nEpoch {epoch + 1}/{total_epochs} "
            f"(seq={current_seq_len}, batch={current_batch_size}, scale={current_scale:.2f})"
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
            dense_targets,
        )

        if val_batches:
            epoch_metrics.update(
                _evaluate(model, val_batches, loss_fn, device_obj, dense_targets)
            )

        epoch_time = time.time() - start_time
        avg_loss = epoch_metrics["avg_loss"]

        history["train_loss"].append(avg_loss)
        history["epoch_times"].append(epoch_time)
        history["metrics"].append(epoch_metrics)

        if save_path:
            _save_history(history, save_path)

        # Keep the best-generalizing weights: the final checkpoint of a
        # run may be past the val minimum
        val_loss = epoch_metrics.get("val_raw_gap")
        if save_path and val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(
                model,
                optimizer,
                tokenizer,
                avg_loss,
                epoch + 1,
                f"{_model_name(save_path)}_best",
                is_final=True,
            )
            print(f"New best val (raw gap {val_loss:.4f}) — saved *_best.pt")

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

        # Early stop: plateau on the held-out metric when a val split
        # exists (the training metric keeps improving during
        # memorization); otherwise on the training raw gap
        stop_metric = epoch_metrics.get("val_raw_gap", raw_gap)
        if stop_metric < best_loss:
            best_loss = stop_metric
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            metric_name = "val" if "val_raw_gap" in epoch_metrics else "train"
            print(
                f"\nEarly stop: {metric_name} loss plateaued "
                f"for {early_stop_patience} epochs"
            )
            break

        # Checkpoint every epoch (overwrite previous to save disk)
        if save_path:
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

    total_elapsed = time.time() - training_start_time
    minutes, seconds = divmod(int(total_elapsed), 60)
    hours, minutes = divmod(minutes, 60)

    print("\nTraining completed!")
    print(f"Final loss: {history['train_loss'][-1]:.4f}")
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    return history


def _model_name(save_path: str) -> str:
    """Model name from a save path ('trained/foo' and 'foo' both -> 'foo')."""
    if "/" in str(save_path) or "\\" in str(save_path):
        return Path(save_path).stem
    return str(save_path)


def _save_history(history: Dict[str, Any], save_path: str) -> None:
    """Persist the training history dict as JSON alongside the checkpoint."""
    path = f"trained/{_model_name(save_path)}_history.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f)


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
    model_name = _model_name(save_path)
    if is_final:
        path = f"trained/{model_name}.pt"
    else:
        path = f"trained/{model_name}_checkpoint.pt"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "config": model.config,
            "model_type": type(getattr(model, "_orig_mod", model)).__name__,
            "tokenizer": tokenizer,
        },
        path,
    )
    label = "Final model" if is_final else "Checkpoint"
    print(f"{label} saved: {path}")
