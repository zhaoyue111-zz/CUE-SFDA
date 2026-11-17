from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "research_v2"
BASE = OUT / "configs" / "E3_raw_low_baseline_resume_gpu1.yaml"


def write_once(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise FileExistsError(f"Refusing to overwrite {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")


def main() -> None:
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    modes = (("E6_fixed50_E3base", "fixed", 0), ("E7_random50_E3base", "random", 1), ("E8_norm50_E3base", "norm", 0), ("E9_cosine50_E3base", "cosine", 1))
    for name, mode, gpu in modes:
        config = deepcopy(base)
        config["experiment"].update({"gpu_id": gpu, "output_dir": str(OUT / name)})
        config["optimizer_routing"] = {
            "enabled": True,
            "mode": mode,
            "group_fraction": 0.5,
            "reference": "one independent frozen-cache High patch per low update",
        }
        path = OUT / "configs" / f"{name}.yaml"
        write_once(path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    manifest = {
        "stage": "D",
        "E5_R0": "E3_raw_low_baseline, all decoder Q/K/V groups update; E4 was excluded from the D baseline because it reduced both mean Dice and Recall versus E3",
        "grouping": "decoder layer × attention projection (Q/K/V) LoRA A/B matrices; 6 layers × 3 projections = 18 groups",
        "active_fraction": 0.5,
        "low_branch_only": True,
        "high_branch": "all Q/K/V updates continue; each low update uses an independent current-model High loss gradient reference from a frozen-cache effective patch",
        "variants": {
            "E6_fixed50_E3base": "fixed first half of sorted Q/K/V groups",
            "E7_random50_E3base": "seeded random half",
            "E8_norm50_E3base": "largest low-gradient norms",
            "E9_cosine50_E3base": "largest cosine alignment to independent High gradients",
        },
        "optimizer_semantics": "unselected parameter gradients are set to None before AdamW.step so momentum and weight decay cannot move them",
        "nonfinite_score_policy": "Nonfinite group scores rank below finite scores; if all groups are nonfinite, sorted group order is the deterministic tie-break",
        "seed": int(base["experiment"]["seed"]),
        "iterations": int(base["method"]["iterations"]),
    }
    write_once(OUT / "stage_d_manifest_E3base.json", json.dumps(manifest, indent=2) + "\n")
    print("Stage D configs ready: E6-E9 on the best C baseline; E5 reuses E3")


if __name__ == "__main__":
    main()
