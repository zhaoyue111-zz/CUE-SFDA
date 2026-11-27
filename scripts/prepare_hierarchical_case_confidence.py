from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def case_id(name: str) -> str:
    return name[:-7] if name.lower().endswith(".nii.gz") else Path(name).stem


def main() -> None:
    source = Path("analysis/hierarchical_uncertainty/stage1_image_level/per_case_scores.csv")
    empty_cache = Path("analysis/hierarchical_dynamic/fixed_pixel_uncertainty")
    destination = Path("analysis/hierarchical_dynamic/image_confidence.csv")
    values: dict[str, dict[str, float]] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] != "I1_Pairwise_Hard" or row["gating"] != "raw":
                continue
            if row["perturbation"] not in {"Image-only", "Prompt-only"}:
                continue
            values.setdefault(row["case_id"], {})[row["perturbation"]] = float(row["confidence"])
    rows = []
    for case, confidence in sorted(values.items()):
        if set(confidence) != {"Image-only", "Prompt-only"}:
            raise ValueError(f"Missing image/prompt confidence for {case}")
        empty_file = empty_cache / f"{case}.npz"
        with np.load(empty_file, allow_pickle=False) as payload:
            empty = bool(payload["original_empty"].item())
        combined = 0.4 * confidence["Image-only"] + 0.6 * confidence["Prompt-only"]
        rows.append({
            "case_id": case,
            "C_image": confidence["Image-only"],
            "C_prompt": confidence["Prompt-only"],
            "C_case_raw": combined,
            "original_prediction_empty": int(empty),
            "C_case": 0.0 if empty else combined,
        })
    if len(rows) != 101:
        raise ValueError(f"Expected 101 Stage-1 cases, found {len(rows)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} cases, empty={sum(row['original_prediction_empty'] for row in rows)} to {destination}")


if __name__ == "__main__":
    main()
