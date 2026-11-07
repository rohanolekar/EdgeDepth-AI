"""Conservatively fine-tune the v2 depth model on NYU training scenes only."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.inference import load_model, save_checkpoint
from src.nyu_dataset import DATASET_ROOT, NYUDepthDataset
from src.nyu_losses import NYURefinementLoss

ROOT = Path(__file__).resolve().parents[1]
V2_CHECKPOINT = ROOT / "models" / "checkpoints" / "depth_refinement_v2_best.pt"
V3_CHECKPOINT = ROOT / "models" / "checkpoints" / "depth_refinement_v3_best.pt"
V3_HISTORY = ROOT / "models" / "checkpoints" / "training_history_v3.json"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _epoch(model, loader, criterion, device: torch.device, optimizer=None,
           grad_clip: float = 1.0) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "l1": 0.0, "gradient": 0.0, "structural": 0.0}
    seen = 0
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            initial = batch["initial_depth"].to(device, non_blocking=True)
            observed = batch["validity_mask"].to(device, non_blocking=True)
            target = batch["ground_truth"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True)
            prediction = model.refine(rgb, initial, observed)
            loss, parts = criterion(prediction, target, valid)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
            count = rgb.shape[0]
            totals["total"] += float(loss.detach()) * count
            for name, value in parts.items():
                totals[name] += float(value) * count
            seen += count
    return {key: value / max(seen, 1) for key, value in totals.items()}


def finetune(*, epochs: int = 30, batch_size: int = 8, learning_rate: float = 1e-5,
             patience: int = 5, seed: int = 1234, workers: int = 0,
             dataset_root: str | Path = DATASET_ROOT,
             initial_checkpoint: str | Path = V2_CHECKPOINT,
             checkpoint_path: str | Path = V3_CHECKPOINT,
             history_path: str | Path = V3_HISTORY,
             device: str | None = None, grad_clip: float = 1.0,
             depth_range: float = 10.0) -> dict:
    """Fine-tune on all train pairs; use val only for loss-based selection."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or patience < 1 or grad_clip <= 0:
        raise ValueError("epochs, batch_size, learning_rate, patience and grad_clip must be positive")
    seed_everything(seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    # The dataset constructor uses only the specified split. Validation items
    # are evaluated under inference_mode and never contribute gradients.
    train_data = NYUDepthDataset(split="train", dataset_root=dataset_root,
                                 depth_range=depth_range, augment=True)
    val_data = NYUDepthDataset(split="val", dataset_root=dataset_root,
                               depth_range=depth_range, augment=False)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=selected_device.type == "cuda",
                              worker_init_fn=_seed_worker, generator=generator)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=selected_device.type == "cuda",
                            worker_init_fn=_seed_worker)

    model, v2_metadata = load_model(initial_checkpoint, device=selected_device)
    v2_depth_range = float(v2_metadata.get("config", {}).get("depth_range", depth_range))
    if not np.isclose(v2_depth_range, depth_range):
        raise ValueError(f"Requested depth_range={depth_range} differs from v2 checkpoint range={v2_depth_range}")
    criterion = NYURefinementLoss(gradient_weight=0.25, structural_weight=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, threshold=1e-4, min_lr=1e-6
    )
    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)

    config = {
        "source_checkpoint": str(initial_checkpoint),
        "dataset": "NYU Depth V2",
        "training_split": "train",
        "validation_split": "val (loss selection only; no gradient updates)",
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "height": 128,
        "width": 128,
        "scale": 4,
        "depth_range": depth_range,
        "initial_interpolation": "bicubic",
        "augmentation": "synchronized horizontal/vertical flips and 90-degree rotations",
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": 1e-4,
        "scheduler": "ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)",
        "gradient_clip_norm": grad_clip,
        "loss_weights": {"masked_l1": 1.0, "edge_weighted_gradient": 0.25,
                         "masked_local_ssim": 0.05},
        "seed": seed,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "patience": patience,
        "device": str(selected_device),
    }
    checkpoint_target, history_target = Path(checkpoint_path), Path(history_path)
    if (checkpoint_target.resolve() == Path(V2_CHECKPOINT).resolve() or
            checkpoint_target.resolve() == Path(initial_checkpoint).resolve()):
        raise ValueError("v3 checkpoint path must be separate from both the v2 checkpoint and initialization path")
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    history_target.parent.mkdir(parents=True, exist_ok=True)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history: list[dict] = []
    started = time.perf_counter()

    for epoch_number in range(1, epochs + 1):
        train_metrics = _epoch(model, train_loader, criterion, selected_device,
                               optimizer=optimizer, grad_clip=grad_clip)
        validation_metrics = _epoch(model, val_loader, criterion, selected_device)
        scheduler.step(validation_metrics["total"])
        history.append({"epoch": epoch_number, "learning_rate": optimizer.param_groups[0]["lr"],
                        "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["total"] < best_loss:
            best_loss, best_epoch, stale = validation_metrics["total"], epoch_number, 0
            save_checkpoint(checkpoint_target, model, epoch=epoch_number,
                            validation_loss=best_loss, config=config)
        else:
            stale += 1

        report = {
            "dataset": "NYU Depth V2 supervised fine-tuning",
            "training_updates_use_split": "train only",
            "validation_use": "loss-based early stopping and checkpoint selection only",
            "device": str(selected_device),
            "gpu_name": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else None,
            "duration_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": (int(torch.cuda.max_memory_allocated(selected_device))
                                      if selected_device.type == "cuda" else None),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "checkpoint": str(checkpoint_target),
            "history_path": str(history_target),
            "history": history,
            "config": config,
        }
        history_target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"epoch {epoch_number}/{epochs}: train={train_metrics['total']:.6f} "
              f"val={validation_metrics['total']:.6f} lr={optimizer.param_groups[0]['lr']:.2e}")
        if stale >= patience:
            break

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--initial-checkpoint", type=Path, default=V2_CHECKPOINT)
    parser.add_argument("--checkpoint", type=Path, default=V3_CHECKPOINT)
    parser.add_argument("--history", type=Path, default=V3_HISTORY)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    result = finetune(epochs=args.epochs, batch_size=args.batch_size,
                      learning_rate=args.learning_rate, patience=args.patience,
                      seed=args.seed, workers=args.workers, dataset_root=args.dataset_root,
                      initial_checkpoint=args.initial_checkpoint, checkpoint_path=args.checkpoint,
                      history_path=args.history, device=args.device)
    print(json.dumps({key: result[key] for key in ("device", "gpu_name", "duration_seconds",
                                                   "best_epoch", "best_validation_loss", "checkpoint")}, indent=2))


if __name__ == "__main__":
    main()
