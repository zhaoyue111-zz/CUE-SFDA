from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "high_voxel_weighting"
CONFIDENCE_CACHE = ROOT / "outputs" / "ablation_consensus_cache"


def load_resolved(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_config(name: str, source: Path) -> Path:
    config = load_resolved(source)
    config.pop("_config_path", None)
    config["experiment"]["output_dir"] = str(OUT / name)
    train = config["train"]
    train["validation_every"] = 0
    train["validate_at_start"] = False
    train["early_stopping_patience"] = 0
    train["high_voxel_weighting"] = {
        "enabled": True,
        "cache_dir": str(CONFIDENCE_CACHE),
        "function": "linear",
        "floor": 0.3,
        "scale": 0.7,
        "power": 1.0,
    }
    path = ROOT / "configs" / f"high_voxel_weighting_{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def main() -> None:
    high_source = OUT.parent / "research_v2" / "E1_high_self_train" / "resolved_config.json"
    full_source = ROOT / "outputs" / "ablation" / "G_dynamic_hard_final" / "resolved_config.json"
    print(write_config("high_only", high_source))
    high_equal = load_resolved(high_source)
    high_equal.pop("_config_path", None)
    high_equal["experiment"]["output_dir"] = str(OUT / "high_only_equal")
    high_equal["train"]["validation_every"] = 0
    high_equal["train"]["validate_at_start"] = False
    high_equal["train"]["early_stopping_patience"] = 0
    high_equal["train"]["high_voxel_weighting"] = {"enabled": False}
    high_equal_path = ROOT / "configs" / "high_voxel_weighting_high_only_equal.yaml"
    high_equal_path.write_text(yaml.safe_dump(high_equal, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(high_equal_path)
    full = load_resolved(full_source)
    full.pop("_config_path", None)
    full["experiment"]["output_dir"] = str(OUT / "full_high_low")
    # Lock the full high/low comparison to the historical fixed best iteration;
    # validation is disabled so no held-out labels influence training or selection.
    full["method"]["iterations"] = 1250
    full["method"]["max_iterations"] = 1250
    full["consensus"]["cache_dir"] = str(OUT / "dynamic_hard_cache")
    train = full["train"]
    train["validation_every"] = 0
    train["validate_at_start"] = False
    train["early_stopping_patience"] = 0
    train["high_voxel_weighting"] = {
        "enabled": True,
        "cache_dir": str(CONFIDENCE_CACHE),
        "function": "linear",
        "floor": 0.3,
        "scale": 0.7,
        "power": 1.0,
    }
    full_path.write_text(yaml.safe_dump(full, sort_keys=False, allow_unicode=True), encoding="utf-8")
    dynamic_cache = OUT / "dynamic_hard_cache"
    if not dynamic_cache.exists():
        shutil.copytree(CONFIDENCE_CACHE, dynamic_cache, copy_function=os.link)
    full_path = ROOT / "configs" / "high_voxel_weighting_full_high_low.yaml"
    full_path.write_text(yaml.safe_dump(full, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(full_path)


if __name__ == "__main__":
    main()
