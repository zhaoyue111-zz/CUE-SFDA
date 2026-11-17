from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--case-ids", type=Path, required=True, help="JSON list or confidence_partition.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--floor", type=float, default=0.3)
    parser.add_argument("--scale", type=float, default=0.7)
    parser.add_argument("--function", choices=("linear", "power"), default="linear")
    parser.add_argument("--power", type=float, default=1.0)
    args = parser.parse_args()

    payload = json.loads(args.case_ids.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        case_ids = payload
    else:
        threshold = float(payload.get("threshold", 0.5))
        case_ids = [name for name, score in payload["scores"].items() if float(score) >= threshold]
    edges = np.linspace(0.0, 1.0, 6)
    confidence_transform = edges if args.function == "linear" else edges**args.power
    weight_edges = args.floor + args.scale * confidence_transform
    confidence_counts = np.zeros(5, dtype=np.int64)
    weight_counts = np.zeros(5, dtype=np.int64)
    confidence_sum = 0.0
    weight_sum = 0.0
    count = 0
    rows = []
    for case_id in sorted(case_ids):
        path = args.cache_dir / "cases" / f"{Path(case_id).name}.npz"
        with np.load(path, allow_pickle=False) as data:
            confidence = np.asarray(data["joint_voxel_confidence"], dtype=np.float32).reshape(-1)
            array_shape = tuple(int(value) for value in data["joint_voxel_confidence"].shape[-3:])
            geometry = json.loads(str(data["geometry_json"].item()))
        if tuple(geometry["padded_shape"]) != array_shape:
            raise ValueError(f"Geometry mismatch for {case_id}: {geometry['padded_shape']} != {array_shape}")
        confidence = np.clip(confidence, 0.0, 1.0)
        transformed = confidence if args.function == "linear" else confidence**args.power
        weight = args.floor + args.scale * transformed
        case_count = int(confidence.size)
        confidence_counts += np.histogram(confidence, bins=edges)[0]
        weight_counts += np.histogram(weight, bins=weight_edges)[0]
        confidence_sum += float(confidence.sum(dtype=np.float64))
        weight_sum += float(weight.sum(dtype=np.float64))
        count += case_count
        rows.append({
            "case": case_id,
            "voxels": case_count,
            "confidence_mean": float(confidence.mean()),
            "confidence_min": float(confidence.min()),
            "confidence_max": float(confidence.max()),
            "weight_mean": float(weight.mean()),
            "weight_min": float(weight.min()),
            "weight_max": float(weight.max()),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_case_confidence_weights.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    bins = []
    for index in range(5):
        low, high = float(edges[index]), float(edges[index + 1])
        bins.append({
            "confidence_interval": f"[{low:.1f},{high:.1f}{']' if index == 4 else ')'}",
            "weight_interval": f"[{weight_edges[index]:.2f},{weight_edges[index + 1]:.2f}{']' if index == 4 else ')'}",
            "confidence_voxels": int(confidence_counts[index]),
            "confidence_fraction": float(confidence_counts[index] / max(count, 1)),
            "weight_voxels": int(weight_counts[index]),
            "weight_fraction": float(weight_counts[index] / max(count, 1)),
        })
    result = {
        "cases": len(case_ids),
        "voxels": count,
        "weight_function": "floor + scale * confidence" if args.function == "linear" else "floor + scale * confidence^power",
        "floor": args.floor,
        "scale": args.scale,
        "power": args.power,
        "confidence_mean": confidence_sum / max(count, 1),
        "weight_mean": weight_sum / max(count, 1),
        "confidence_weight_bins": bins,
        "padding_note": "Counts include padded voxels because padded regions are present in training patches.",
    }
    (args.output_dir / "confidence_weight_distribution.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
