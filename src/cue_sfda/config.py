from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {config_path}")
    config = deepcopy(config)
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {"experiment", "data", "model", "method", "train", "evaluation"}
    missing = required.difference(config)
    if missing:
        raise KeyError(f"Missing configuration sections: {sorted(missing)}")
    iterations = config["method"].get("iterations", config["method"].get("max_iterations", 0))
    if int(iterations) <= 0:
        raise ValueError("method.iterations/max_iterations must be positive")
    gpu_id = config["experiment"].get("gpu_id")
    if gpu_id is not None and int(gpu_id) < 0:
        raise ValueError("experiment.gpu_id must be non-negative")
    patch_size = tuple(int(value) for value in config["data"]["patch_size"])
    if len(patch_size) != 3:
        raise ValueError("data.patch_size must contain three values")
    if any(value <= 0 or value % 32 != 0 for value in patch_size):
        raise ValueError("Every patch dimension must be positive and divisible by 32")
    deepest_shape = tuple(value // 32 for value in patch_size)
    deepest_voxels = deepest_shape[0] * deepest_shape[1] * deepest_shape[2]
    if deepest_voxels <= 1:
        raise ValueError(
            "data.patch_size is too small for VoxTell InstanceNorm3d"
        )
    if int(config["method"]["num_views"]) < 2:
        raise ValueError("method.num_views must be at least two")
    quality_threshold = float(config["method"].get("quality_threshold", 0.5))
    if not 0.0 <= quality_threshold <= 1.0:
        raise ValueError("method.quality_threshold must be in [0, 1]")
    sampling_probability = float(
        config["method"].get("high_quality_sampling_probability", 0.5)
    )
    if not 0.0 <= sampling_probability <= 1.0:
        raise ValueError(
            "method.high_quality_sampling_probability must be in [0, 1]"
        )
    if int(config["method"].get("min_case_pseudo_foreground_voxels", 1)) < 1:
        raise ValueError(
            "method.min_case_pseudo_foreground_voxels must be positive"
        )
    routing_mode = str(config["method"].get("routing_mode", "original"))
    if routing_mode not in {"original", "image", "joint", "joint_gate"}:
        raise ValueError("method.routing_mode must be original/image/joint/joint_gate")
    stage2_mode = str(config.get("stage2", {}).get("mode", "none"))
    if stage2_mode not in {
        "none", "global", "local", "prior", "prior_fixed", "weighted_prior",
        "clustered_prior", "clustered_boundary", "full",
    }:
        raise ValueError("invalid stage2.mode")
    tile_step_size = float(config["train"].get("tile_step_size", 0.5))
    if not 0.0 < tile_step_size <= 1.0:
        raise ValueError("train.tile_step_size must be in (0, 1]")
    if int(config["train"].get("validation_every", 0)) < 0:
        raise ValueError("train.validation_every cannot be negative")
    if float(config["train"].get("high_foreground_weight", 1.0)) < 1.0:
        raise ValueError("train.high_foreground_weight must be at least 1")
    if float(config["train"].get("low_foreground_weight", 1.0)) < 1.0:
        raise ValueError("train.low_foreground_weight must be at least 1")
    consensus = config.get("consensus", {})
    mode = str(consensus.get("mode", "none"))
    valid_modes = {
        "none", "image", "prompt", "joint", "joint_language",
        "dynamic_hard", "dynamic_ema", "dynamic_hard_language",
        "selective_dynamic", "selective_dynamic_language",
    }
    if mode not in valid_modes:
        raise ValueError(f"invalid consensus.mode: {mode}")
    prompts = consensus.get("prompts", config["method"].get("routing_prompts", []))
    if mode != "none" and not prompts:
        raise ValueError("consensus.prompts must be non-empty when consensus is enabled")
    if float(consensus.get("lambda_consensus", 0.05)) < 0:
        raise ValueError("consensus.lambda_consensus cannot be negative")
    if float(consensus.get("lambda_language", 0.01)) < 0:
        raise ValueError("consensus.lambda_language cannot be negative")
    high_selection = config.get("high_selection", {})
    high_selection_mode = str(high_selection.get("mode", "none"))
    if high_selection_mode not in {"none", "semantic_subspace"}:
        raise ValueError("high_selection.mode must be none or semantic_subspace in this repository")
    if high_selection_mode == "semantic_subspace":
        if not config.get("method", {}).get("routing_assignments_csv"):
            raise ValueError("CSS requires a frozen method.routing_assignments_csv")
        if int(high_selection.get("reference_count", 8)) < 4:
            raise ValueError("CSS requires at least four reference cases")
        if int(high_selection.get("reference_refresh_interval", 100)) < 1:
            raise ValueError("high_selection.reference_refresh_interval must be positive")
        energy = float(high_selection.get("spectral_energy", 0.9))
        if not 0.0 < energy <= 1.0:
            raise ValueError("high_selection.spectral_energy must be in (0, 1]")
        if int(high_selection.get("tokens_per_case", 256)) < 1:
            raise ValueError("high_selection.tokens_per_case must be positive")
        if float(high_selection.get("robust_mad_multiplier", 3.0)) < 0.0:
            raise ValueError("high_selection.robust_mad_multiplier cannot be negative")
    patience = int(config["train"].get("early_stopping_patience", 10))
    if patience < 0:
        raise ValueError("train.early_stopping_patience cannot be negative")


def apply_overrides(config: dict[str, Any], args: Any) -> dict[str, Any]:
    if getattr(args, "prompt", None):
        config["model"]["prompt"] = args.prompt
    if getattr(args, "output_dir", None):
        config["experiment"]["output_dir"] = args.output_dir
    if getattr(args, "device", None):
        config["model"]["device"] = args.device
        config["experiment"].pop("gpu_id", None)
    if getattr(args, "gpu_id", None) is not None:
        config["experiment"]["gpu_id"] = int(args.gpu_id)
    if getattr(args, "iterations", None) is not None:
        config["method"]["iterations"] = int(args.iterations)
        config["method"]["max_iterations"] = int(args.iterations)
    if getattr(args, "max_iterations", None) is not None:
        config["method"]["iterations"] = int(args.max_iterations)
        config["method"]["max_iterations"] = int(args.max_iterations)
    if getattr(args, "calibration_cases", None) is not None:
        config["method"]["calibration_cases"] = int(args.calibration_cases)
    if getattr(args, "routing_mode", None):
        config["method"]["routing_mode"] = str(args.routing_mode)
    if getattr(args, "stage2_mode", None):
        config.setdefault("stage2", {})["mode"] = str(args.stage2_mode)
    if getattr(args, "stage2_top_k", None) is not None:
        config.setdefault("stage2", {})["top_k"] = int(args.stage2_top_k)
    if getattr(args, "consensus_mode", None):
        config.setdefault("consensus", {})["mode"] = str(args.consensus_mode)
    if getattr(args, "consensus_cache", None):
        config.setdefault("consensus", {})["cache_dir"] = str(args.consensus_cache)
    validate_config(config)
    return config
