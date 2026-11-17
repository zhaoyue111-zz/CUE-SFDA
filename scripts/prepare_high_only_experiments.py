from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import yaml

from cue_sfda.data import load_split


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "research_v2"
BASE_CONFIG = ROOT / "configs" / "response_routing_liver_cancer.yaml"
ROUTING = ROOT / "outputs" / "response_routing" / "C_joint" / "routing.csv"
GT_DIR = Path("/data/hilab/datasets/WCH_MRI145/DWI/labels")


def write_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to overwrite existing protocol file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    train, test = load_split(base["data"]["image_dir"], base["data"]["split_json"])
    train_ids = {case.name for case in train}
    test_ids = {case.name for case in test}
    with ROUTING.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    high_ids = sorted(row["case_id"] for row in rows if row["assigned_branch"] == "high")
    if len(high_ids) != 59 or len(set(high_ids)) != 59:
        raise ValueError(f"Expected 59 unique C_joint High cases; found {len(high_ids)}")
    if not set(high_ids) <= train_ids:
        raise ValueError("C_joint High partition includes cases outside the 101-case train split")
    if set(high_ids) & test_ids:
        raise ValueError("C_joint High cases overlap the 44-case held-out split")

    split = {
        "train_cases": [str(Path(base["data"]["image_dir"]) / name) for name in high_ids],
        "test_cases": [str(case.image_path) for case in test],
    }
    split_path = OUT / "high59_split.json"
    write_once(split_path, json.dumps(split, indent=2) + "\n")
    partition_path = OUT / "high59_partition.json"
    partition = {"threshold": 0.5, "scores": {name: 1.0 for name in high_ids}, "foreground_counts": {}}
    write_once(partition_path, json.dumps(partition, indent=2) + "\n")

    for name, gpu_id, relation in (("E1_high_self_train", 0, False), ("H1v1_high_relation", 1, True)):
        config = deepcopy(base)
        config["experiment"].update({"gpu_id": gpu_id, "output_dir": str(OUT / name)})
        config["data"]["split_json"] = str(split_path)
        config["model"]["prompt"] = "liver cancer"
        # The plan's trainable set is decoder Q/K/V LoRA only.
        config["model"].pop("lora_linear_targets", None)
        config["method"].update({"iterations": 500, "max_iterations": 500})
        config["train"].update({
            "high_pseudo_cache_dir": str(OUT / "shared_high_pseudo_cache"),
            "validation_every": 0,
            "validate_at_start": False,
            "early_stopping_patience": 0,
            "save_every": 100,
            "log_every": 10,
        })
        config["consensus"] = {"mode": "none"}
        config["stage2"] = {"mode": "none"}
        config["high_relation"] = {
            "enabled": relation,
            "weight": 0.1 if relation else 0.0,
            "top_k": 5,
            "temperature": 0.1,
        }
        config_path = OUT / "configs" / f"{name}.yaml"
        write_once(config_path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

    manifest = {
        "protocol": "research_v2_stage_A_B_initial",
        "routing_csv": str(ROUTING),
        "base_config": str(BASE_CONFIG),
        "prompt": "liver cancer",
        "high_case_count": len(high_ids),
        "source_train_count": len(train_ids),
        "heldout_count": len(test_ids),
        "high_case_ids": high_ids,
        "fixed_iterations": 500,
        "seed": int(base["experiment"]["seed"]),
        "trainable_parameters": "decoder Q/K/V LoRA only",
        "initial_pseudo_target": "shared frozen-source random-walk cache",
        "h1v1": {"weight": 0.1, "top_k": 5, "temperature": 0.1, "self_excluded": True},
        "evaluation_caveat": "The 44-case split has been reused for prior method/checkpoint selection; this run uses a fixed final iteration and is exploratory labeled development evaluation, not an untouched final test.",
    }
    write_once(OUT / "protocol_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Protocol ready: High={len(high_ids)} / train={len(train_ids)} / heldout={len(test_ids)}")
    print(f"Split: {split_path}")
    print(f"Configurations: {OUT / 'configs'}")


if __name__ == "__main__":
    main()
