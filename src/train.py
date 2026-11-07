"""Train the lightweight residual depth-refinement CNN on synthetic scenes."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.depth_refinement_cnn import DepthRefinementCNN
from src.dataset import SyntheticDataConfig, SyntheticDepthDataset
from src.inference import save_checkpoint
from src.losses import DepthReconstructionLoss

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "checkpoints" / "depth_refinement_best.pt"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    sums = {"total": 0.0, "l1": 0.0, "gradient": 0.0}
    count = 0
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            initial = batch["initial_depth"].to(device, non_blocking=True)
            validity = batch["validity_mask"].to(device, non_blocking=True)
            target = batch["ground_truth"].to(device, non_blocking=True)
            valid_gt = batch["valid_mask"].to(device, non_blocking=True)
            prediction = model.refine(rgb, initial, validity)
            loss, parts = criterion(prediction, target, valid_gt)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch_size = rgb.shape[0]
            sums["total"] += float(loss.detach()) * batch_size
            sums["l1"] += float(parts["l1"]) * batch_size
            sums["gradient"] += float(parts["gradient"]) * batch_size
            count += batch_size
    return {key: value / max(count, 1) for key, value in sums.items()}


def train(*, epochs=3, batch_size=4, train_samples=32, validation_samples=8,
          height=64, width=64, scale=2, learning_rate=1e-3, seed=1234,
          noise_std=.03, missing_rate=.05, patience=5, workers=0,
          checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
          history_path: str | Path | None = None,
          device: str | None = None, base_channels=16) -> dict:
    if epochs < 1 or batch_size < 1 or train_samples < 1 or validation_samples < 1:
        raise ValueError("epochs, batch size and dataset sizes must be positive")
    seed_everything(seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cfg = dict(height=height, width=width, scale=scale, noise_std=noise_std,
               missing_rate=missing_rate, max_depth=10.0)
    train_data = SyntheticDepthDataset(SyntheticDataConfig(samples=train_samples, seed=seed, **cfg), split="train")
    val_data = SyntheticDepthDataset(SyntheticDataConfig(samples=validation_samples, seed=seed, **cfg), split="validation")
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers,
                              pin_memory=selected_device.type == "cuda", generator=loader_generator)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers,
                            pin_memory=selected_device.type == "cuda")
    model = DepthRefinementCNN(base_channels=base_channels).to(selected_device)
    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)
    criterion = DepthReconstructionLoss(gradient_weight=.25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    config = {"epochs_requested": epochs, "batch_size": batch_size,
              "train_samples": train_samples, "validation_samples": validation_samples,
              "height": height, "width": width, "scale": scale,
              "noise_std": noise_std, "missing_rate": missing_rate, "seed": seed,
              "learning_rate": learning_rate, "base_channels": base_channels,
              "gradient_weight": .25,
              "loss_note": "Retained the v1 gradient weight to isolate the effect of increased training data and updates.",
              "depth_range": 10.0, "device": str(selected_device),
              "dataset_type": "procedural synthetic scenes"}
    history = []
    best = float("inf")
    best_epoch = 0
    stale = 0
    target = Path(checkpoint_path)
    started_at = time.perf_counter()
    for epoch in range(1, epochs+1):
        train_metrics = _epoch(model, train_loader, criterion, selected_device, optimizer)
        val_metrics = _epoch(model, val_loader, criterion, selected_device)
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        if val_metrics["total"] < best:
            best, best_epoch, stale = val_metrics["total"], epoch, 0
            save_checkpoint(target, model, epoch=epoch, validation_loss=best, config=config)
        else:
            stale += 1
        print(f"epoch {epoch}/{epochs}: train={train_metrics['total']:.6f} val={val_metrics['total']:.6f}")
        if patience > 0 and stale >= patience:
            break
    duration_seconds = time.perf_counter() - started_at
    report = {"device": str(selected_device),
              "gpu_name": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else None,
              "duration_seconds": duration_seconds,
              "peak_gpu_memory_bytes": (int(torch.cuda.max_memory_allocated(selected_device))
                                        if selected_device.type == "cuda" else None),
              "best_epoch": best_epoch, "best_validation_loss": best,
              "checkpoint": str(target), "history": history, "config": config}
    history_target = Path(history_path) if history_path else target.with_name("training_history.json")
    report["history_path"] = str(history_target)
    history_target.parent.mkdir(parents=True, exist_ok=True)
    history_target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--validation-samples", type=int, default=8)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--scale", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    report = train(epochs=args.epochs, batch_size=args.batch_size, train_samples=args.train_samples,
                   validation_samples=args.validation_samples, height=args.height, width=args.width,
                   scale=args.scale, learning_rate=args.learning_rate, seed=args.seed,
                   patience=args.patience, workers=args.workers, checkpoint_path=args.checkpoint,
                   history_path=args.history,
                   device=args.device)
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
