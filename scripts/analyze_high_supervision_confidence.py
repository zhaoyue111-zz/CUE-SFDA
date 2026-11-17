from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cue_sfda.data import TargetVolumeStore, load_split
from cue_sfda.method.consensus import ConsensusCache
from cue_sfda.method.high_pseudo_cache import HighPseudoCache


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit confidence on effective high-branch supervision patches.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--confidence-cache", type=Path, required=True)
    parser.add_argument("--pseudo-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    partition = json.loads(args.partition.read_text(encoding="utf-8"))
    if isinstance(partition, list):
        case_ids = [str(name) for name in partition]
    else:
        threshold = float(partition.get("threshold", 0.5))
        case_ids = [name for name, score in partition["scores"].items() if float(score) >= threshold]
    train, _ = load_split(config["data"]["image_dir"], config["data"]["split_json"])
    cases = {case.name: case for case in train}
    confidence_cache = ConsensusCache(args.confidence_cache)
    pseudo_cache = HighPseudoCache(args.pseudo_cache, expected_cases=case_ids)
    store = TargetVolumeStore(tuple(int(v) for v in config["data"]["patch_size"]))
    edges = np.linspace(0.0, 1.0, 6)
    categories = {name: np.zeros(5, dtype=np.int64) for name in ("all", "foreground", "background")}
    weighted_sums = {name: 0.0 for name in categories}
    weight_sums = {name: 0.0 for name in categories}
    voxel_counts = {name: 0 for name in categories}
    effective_patches = skipped_patches = 0
    foreground_voxels = background_voxels = 0
    min_foreground = int(config["method"].get("min_pseudo_foreground_voxels", 8))
    floor, scale = 0.3, 0.7

    for case_id in case_ids:
        case = cases[case_id]
        confidence = confidence_cache.load_case(case)["joint_voxel_confidence"]
        _, slicers = store.sliding_window(case, float(config["train"].get("tile_step_size", 0.5)))
        pseudo_cache._load_case(case_id)
        targets = pseudo_cache._targets[case_id]
        counts = pseudo_cache._counts[case_id]
        if len(slicers) != targets.shape[0]:
            raise ValueError(f"Pseudo-label patch grid mismatch for {case_id}")
        for patch_index, slicer in enumerate(slicers):
            if int(counts[patch_index, 1]) < min_foreground:
                skipped_patches += 1
                continue
            effective_patches += 1
            c = np.asarray(confidence[(slice(None), *slicer)][0], dtype=np.float32)
            y = np.asarray(targets[patch_index], dtype=np.float32)
            if c.shape != y.shape:
                raise ValueError(f"Confidence/target patch mismatch for {case_id} patch {patch_index}")
            for name, mask in (("all", np.ones(y.shape, dtype=bool)), ("foreground", y >= 0.5), ("background", y < 0.5)):
                values = np.clip(c[mask], 0.0, 1.0)
                if values.size == 0:
                    continue
                weights = floor + scale * values
                categories[name] += np.histogram(values, bins=edges)[0]
                weighted_sums[name] += float(values.sum(dtype=np.float64))
                weight_sums[name] += float(weights.sum(dtype=np.float64))
                voxel_counts[name] += int(values.size)
            foreground_voxels += int((y >= 0.5).sum())
            background_voxels += int((y < 0.5).sum())

    bins = []
    for index in range(5):
        bins.append({
            "confidence_interval": f"[{edges[index]:.1f},{edges[index + 1]:.1f}{']' if index == 4 else ')'}",
            **{f"{name}_voxels": int(categories[name][index]) for name in categories},
        })
    result = {
        "high_cases": len(case_ids),
        "effective_patches": effective_patches,
        "skipped_patches": skipped_patches,
        "effective_foreground_voxels": foreground_voxels,
        "effective_background_voxels": background_voxels,
        "weight_function": "0.3 + 0.7 * confidence",
        "groups": {
            name: {
                "voxel_count": voxel_counts[name],
                "confidence_mean": weighted_sums[name] / max(voxel_counts[name], 1),
                "weight_mean": weight_sums[name] / max(voxel_counts[name], 1),
            }
            for name in categories
        },
        "confidence_bins_on_effective_training_patches": bins,
        "overlap_note": "Counts are patch exposures, including repeated voxels from overlapping sliding windows, and exclude skipped patches.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
