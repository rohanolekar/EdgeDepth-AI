"""Checkpoint loading and depth-refinement inference helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from models.depth_refinement_cnn import DepthRefinementCNN


def save_checkpoint(path: str | Path, model: DepthRefinementCNN, *, epoch: int,
                    validation_loss: float, config: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "epoch": int(epoch),
                "validation_loss": float(validation_loss), "config": config,
                "model": {"base_channels": model.enc1.layers[0].out_channels,
                          "max_residual": model.max_residual}}, target)


def load_model(path: str | Path, *, device: str | torch.device = "cpu"):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    model_config = checkpoint.get("model", {})
    model = DepthRefinementCNN(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def refine_batch(model: DepthRefinementCNN, rgb: torch.Tensor,
                 initial_depth: torch.Tensor, validity_mask: torch.Tensor) -> torch.Tensor:
    """Run model refinement; inputs and output use normalized [0, 1] units."""
    return model.refine(rgb, initial_depth, validity_mask)
