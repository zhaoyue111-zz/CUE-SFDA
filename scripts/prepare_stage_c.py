from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import yaml

from cue_sfda.data import load_split


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "research_v2"
BASE = ROOT / "configs" / "response_routing_liver_cancer.yaml"
ROUTING = ROOT / "outputs" / "response_routing" / "C_joint" / "routing.csv"


def write_once(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def main() -> None:
    config = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    train, test = load_split(config["data"]["image_dir"], config["data"]["split_json"])
    train_ids = {case.name for case in train}
    test_ids = {case.name for case in test}
    with ROUTING.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    routes = {row["case_id"]: row["assigned_branch"] for row in rows}
    if set(routes) != train_ids or set(routes.values()) != {"high", "low"}:
        raise ValueError("C_joint routing does not exactly cover the original 101-case train split")
    high_ids = sorted(name for name, branch in routes.items() if branch == "high")
    low_ids = sorted(name for name, branch in routes.items() if branch == "low")
    if (len(high_ids), len(low_ids), len(test_ids)) != (59, 42, 44):
        raise ValueError(f"Unexpected counts: High={len(high_ids)} Low={len(low_ids)} heldout={len(test_ids)}")

    partition = {
        "threshold": 0.5,
        "scores": {name: (1.0 if name in set(high_ids) else 0.0) for name in sorted(train_ids)},
        "foreground_counts": {},
    }
    partition_path = OUT / "full101_C_joint_partition.json"
    write_once(partition_path, json.dumps(partition, indent=2) + "\n")

    for name, gpu_id, enabled in (("E3_raw_low_baseline", 0, False), ("E4_high_memory_low_relation", 1, True)):
        cfg = deepcopy(config)
        cfg["experiment"].update({"gpu_id": gpu_id, "output_dir": str(OUT / name)})
        cfg["model"]["prompt"] = "liver cancer"
        cfg["model"].pop("lora_linear_targets", None)
        cfg["method"].update({
            "iterations": 1000,
            "max_iterations": 1000,
            "routing_uncertainty_csv": None,
            "routing_reference_partition": None,
        })
        cfg["train"].update({
            "high_pseudo_cache_dir": str(OUT / "shared_high_pseudo_cache"),
            "high_pseudo_cache_case_ids": high_ids,
            "low_raw_frozen_source": True,
            "validation_every": 0,
            "validate_at_start": False,
            "early_stopping_patience": 0,
            "save_every": 100,
            "log_every": 10,
        })
        cfg["consensus"] = {"mode": "none"}
        cfg["stage2"] = {"mode": "none"}
        cfg["high_relation"] = {"enabled": False, "weight": 0.0}
        cfg["low_relation"] = {
            "enabled": enabled,
            "weight": 0.1 if enabled else 0.0,
            "gradient_checkpointing": enabled,
        }
        path = OUT / "configs" / f"{name}.yaml"
        write_once(path, yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    manifest = {
        "stage": "C",
        "routing": str(ROUTING),
        "train_cases": len(train_ids),
        "high_cases": len(high_ids),
        "low_cases": len(low_ids),
        "heldout_cases": len(test_ids),
        "prompt": "liver cancer",
        "iterations": 1000,
        "seed": int(config["experiment"]["seed"]),
        "E3_E4_shared": ["C_joint routing", "frozen-source High pseudo-label cache", "raw-image low branch", "Q/K/V LoRA only", "same optimizer and step budget"],
        "low_baseline_deviation": "Repository's existing low branch uses augmented-view teacher/student KL. This Stage C control uses an explicit opt-in raw-image frozen-source Bernoulli-KL baseline to follow 1.md's no-view rule; default trainer behavior is unchanged.",
        "E4_only_change": "On low-query foreground voxels from the frozen source soft target, add 0.1 * mean(1 - max cosine(student raw D5 feature, frozen High foreground prototype bank)). No GT enters this loss.",
        "E4_engineering": "Activation checkpointing is enabled for the student feature forward to fit the optional relation graph within 11 GB GPU memory; it preserves the forward loss and gradient.",
        "gt_used_for_training_or_selection": False,
    }
    write_once(OUT / "stage_c_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Stage C ready: High={len(high_ids)} Low={len(low_ids)} heldout={len(test_ids)}")


if __name__ == "__main__":
    main()
