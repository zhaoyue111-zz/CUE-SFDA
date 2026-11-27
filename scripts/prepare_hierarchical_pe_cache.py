from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cue_sfda.config import load_config
from cue_sfda.data import TargetVolumeStore, load_split


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    return -(probability * np.log2(probability) + (1.0 - probability) * np.log2(1.0 - probability))


def case_id(name: str) -> str:
    return name[:-7] if name.lower().endswith(".nii.gz") else Path(name).stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed Stage-1 predictive-entropy reliability maps")
    parser.add_argument("--config", type=Path, default=Path("configs/hierarchical_m1.yaml"))
    parser.add_argument("--probability-dir", type=Path, default=Path("analysis/hierarchical_uncertainty/stage1_image_level/probabilities"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/hierarchical_dynamic/fixed_pixel_uncertainty"))
    args = parser.parse_args()
    config = load_config(args.config)
    train_cases, _ = load_split(config["data"]["image_dir"], config["data"]["split_json"])
    store = TargetVolumeStore(tuple(config["data"]["patch_size"]), cache_size=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for case in sorted(train_cases, key=lambda row: row.name):
        key = case_id(case.name)
        source = args.probability_dir / f"{key}.npz"
        target = args.output_dir / f"{key}.npz"
        if not source.is_file():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as payload:
            image_views = payload["image_views"].astype(np.float32, copy=False)
            prompt_views = payload["prompt_views"].astype(np.float32, copy=False)
            original = payload["original"].astype(np.float32, copy=False)
            source_metadata = json.loads(str(payload["metadata"].item()))
        image_entropy = binary_entropy(image_views.mean(axis=0))
        prompt_entropy = binary_entropy(prompt_views.mean(axis=0))
        reliability = np.clip(1.0 - 0.5 * (image_entropy + prompt_entropy), 0.0, 1.0)
        _, _, geometry = store.sliding_window_with_geometry(case, float(config["train"]["tile_step_size"]))
        if tuple(reliability.shape) != tuple(geometry["roi_shape"]):
            raise ValueError(f"Stage-1 ROI geometry mismatch for {case.name}: {reliability.shape} vs {geometry['roi_shape']}")
        padded = np.zeros((1, *geometry["padded_shape"]), dtype=np.float32)
        padded[(slice(None), *geometry["roi_slicer"])] = reliability[None]
        original_empty = not bool(np.any(original >= float(config["evaluation"]["threshold"])))
        metadata = {
            "case_id": key,
            "source_probability_file": str(source.resolve()),
            "source_model_checkpoint": source_metadata["model_checkpoint"],
            "entropy_base": "log2",
            "fusion": "0.5*H(mean(image_views))+0.5*H(mean(prompt_views))",
            "weight": "1-U_pixel",
            "roi_shape": list(geometry["roi_shape"]),
            "padded_shape": list(geometry["padded_shape"]),
            "roi_slicer": [[int(s.start or 0), int(s.stop)] for s in geometry["roi_slicer"]],
            "original_prediction_empty": original_empty,
        }
        temporary = target.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, weight=padded, original_empty=np.asarray(original_empty), metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
        temporary.replace(target)
        manifest.append(metadata)
        print(f"cached {key} empty={original_empty} shape={padded.shape}", flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"completed cases={len(manifest)} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
