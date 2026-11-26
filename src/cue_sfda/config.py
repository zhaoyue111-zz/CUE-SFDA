from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from cue_sfda.method.liver_adaptation import EXPERIMENTS


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
    if int(config["method"]["iterations"]) <= 0:
        raise ValueError("method.iterations must be positive")
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
    if routing_mode not in {
        "original", "image", "joint", "joint_gate", "joint_empty_to_low"
    }:
        raise ValueError(
            "method.routing_mode must be original/image/joint/joint_gate/"
            "joint_empty_to_low"
        )
    dynamic_hard = config.get("dynamic_hard", {})
    dynamic_mode = str(dynamic_hard.get("mode", "none"))
    if dynamic_mode not in {"none", "dynamic_hard"}:
        raise ValueError("dynamic_hard.mode must be none or dynamic_hard")
    if dynamic_mode == "dynamic_hard":
        if not dynamic_hard.get("cache_dir"):
            raise ValueError("dynamic_hard.cache_dir is required")
        prompts = dynamic_hard.get("prompts", [])
        standard_prompt = str(dynamic_hard.get("standard_prompt", ""))
        if not prompts or standard_prompt not in {str(prompt) for prompt in prompts}:
            raise ValueError(
                "dynamic_hard.prompts must include dynamic_hard.standard_prompt"
            )
        if float(dynamic_hard.get("lambda_consensus", 0.05)) < 0.0:
            raise ValueError("dynamic_hard.lambda_consensus cannot be negative")
        if int(dynamic_hard.get("max_refresh_cycles", 1)) < 0:
            raise ValueError("dynamic_hard.max_refresh_cycles cannot be negative")
        refresh_step = float(dynamic_hard.get("refresh_tile_step_size", 1.0))
        if not 0.0 < refresh_step <= 1.0:
            raise ValueError("dynamic_hard.refresh_tile_step_size must be in (0, 1]")
        if routing_mode != "joint_empty_to_low":
            raise ValueError(
                "the M1 Dynamic-hard config requires joint_empty_to_low routing"
            )
        if not int(config["method"].get("routing_high_cases", 0)):
            raise ValueError("dynamic-hard M1 requires method.routing_high_cases")
    high_pseudo_target = str(config["method"].get("high_pseudo_target", "source"))
    if high_pseudo_target not in {"source", "prompt_consensus"}:
        raise ValueError("method.high_pseudo_target must be source or prompt_consensus")
    if high_pseudo_target == "prompt_consensus" and dynamic_mode != "dynamic_hard":
        raise ValueError("prompt_consensus High targets require the Dynamic-hard cache")
    tile_step_size = float(config["train"].get("tile_step_size", 0.5))
    if not 0.0 < tile_step_size <= 1.0:
        raise ValueError("train.tile_step_size must be in (0, 1]")
    if int(config["train"].get("validation_every", 0)) < 0:
        raise ValueError("train.validation_every cannot be negative")
    high_selection = config.get("high_selection", {})
    high_selection_mode = str(high_selection.get("mode", "none"))
    if high_selection_mode not in {
        "none", "gradient_agreement", "cross_case_validation",
        "semantic_subspace", "semantic_update_validation",
    }:
        raise ValueError(f"invalid high_selection.mode: {high_selection_mode}")
    if high_selection_mode != "none" and str(config.get("dynamic_hard", {}).get("mode", "none")) != "dynamic_hard":
        raise ValueError("High Selection modes require the M1 Dynamic-hard training path")
    if int(high_selection.get("reference_count", 8)) < 2:
        raise ValueError("high_selection.reference_count must be at least 2")
    if high_selection_mode in {"semantic_subspace", "semantic_update_validation"} and int(high_selection.get("reference_count", 8)) < 4:
        raise ValueError("CSS/SUV require high_selection.reference_count of at least 4")
    if int(high_selection.get("reference_refresh_interval", 100)) < 1:
        raise ValueError("high_selection.reference_refresh_interval must be positive")
    significance = float(high_selection.get("significance_level", 0.05))
    if not 0.0 < significance < 0.5:
        raise ValueError("high_selection.significance_level must be in (0, 0.5)")
    energy = float(high_selection.get("spectral_energy", 0.9))
    if not 0.0 < energy <= 1.0:
        raise ValueError("high_selection.spectral_energy must be in (0, 1]")
    if int(high_selection.get("tokens_per_case", 256)) < 1:
        raise ValueError("high_selection.tokens_per_case must be positive")
    if float(high_selection.get("robust_mad_multiplier", 3.0)) < 0.0:
        raise ValueError("high_selection.robust_mad_multiplier cannot be negative")
    if int(config["train"].get("early_stopping_patience", 0)) < 0:
        raise ValueError("train.early_stopping_patience cannot be negative")
    if float(config["train"].get("high_foreground_weight", 1.0)) < 1.0:
        raise ValueError("train.high_foreground_weight must be at least 1")
    if float(config["train"].get("low_foreground_weight", 1.0)) < 1.0:
        raise ValueError("train.low_foreground_weight must be at least 1")
    liver = config.get("liver_adaptation", {})
    liver_experiment = str(liver.get("experiment", "baseline"))
    if liver_experiment not in EXPERIMENTS:
        raise ValueError(
            "liver_adaptation.experiment must be baseline or one of E0-E5"
        )
    if liver and not liver.get("teacher_cache_dir"):
        raise ValueError("liver_adaptation.teacher_cache_dir is required")
    volume = liver.get("volume_constraint", {})
    if float(volume.get("margin", 0.1)) < 0.0 or float(volume.get("margin", 0.1)) >= 1.0:
        raise ValueError("liver_adaptation.volume_constraint.margin must be in [0, 1)")
    if float(volume.get("weight", 1.0)) < 0.0:
        raise ValueError("liver_adaptation.volume_constraint.weight cannot be negative")
    if float(volume.get("shrink_weight", 1.0)) < 0.0:
        raise ValueError("liver_adaptation.volume_constraint.shrink_weight cannot be negative")
    if float(liver.get("remote_background", {}).get("distance_mm", 15.0)) <= 0:
        raise ValueError("liver_adaptation.remote_background.distance_mm must be positive")
    if float(liver.get("regions", {}).get("boundary_width_mm", 5.0)) <= 0:
        raise ValueError("liver_adaptation.regions.boundary_width_mm must be positive")
    if float(liver.get("anchor", {}).get("weight", 0.0)) < 0:
        raise ValueError("liver_adaptation.anchor.weight cannot be negative")


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
    if getattr(args, "calibration_cases", None) is not None:
        config["method"]["calibration_cases"] = int(args.calibration_cases)
    if getattr(args, "routing_mode", None):
        config["method"]["routing_mode"] = str(args.routing_mode)
    validate_config(config)
    return config
