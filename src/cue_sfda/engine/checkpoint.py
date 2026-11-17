from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from cue_sfda.modeling.lora import load_lora_state_dict, lora_state_dict


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    config: dict,
    confidence_scores: dict[str, float],
    confidence_threshold: float,
    best_dice: float | None = None,
    best_iteration: int | None = None,
    trainer_generator_state: torch.Tensor | None = None,
    view_generator_state: torch.Tensor | None = None,
    view_python_generator_state: object | None = None,
    scaler: object | None = None,
    trainer_state: dict | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 3,
        "iteration": int(iteration),
        "lora_state": lora_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": None if scaler is None else scaler.state_dict(),
        "confidence_scores": dict(confidence_scores),
        "confidence_threshold": float(confidence_threshold),
        "best_dice": None if best_dice is None else float(best_dice),
        "best_iteration": (
            None if best_iteration is None else int(best_iteration)
        ),
        "config": config,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "trainer_generator": trainer_generator_state,
            "view_generator": view_generator_state,
            "view_python_generator": view_python_generator_state,
        },
    }
    if trainer_state is not None:
        payload["trainer_state"] = dict(trainer_state)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(output_path)
    return output_path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: object | None = None,
) -> dict:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    load_lora_state_dict(model, checkpoint["lora_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler_state = checkpoint.get("scaler_state")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    return checkpoint
