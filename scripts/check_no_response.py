#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization

from cue_sfda.config import load_config
from cue_sfda.data import load_split
from cue_sfda.evaluation import predict_volume
from cue_sfda.modeling import build_runtime
from cue_sfda.utils import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--routing-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    config["experiment"]["gpu_id"] = args.gpu_id
    seed_everything(int(config["experiment"]["seed"]))
    train_cases, _ = load_split(config["data"]["image_dir"], config["data"]["split_json"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = build_runtime(config, args.output_dir / "prompt_cache")
    reader, normalizer = NibabelIOWithReorient(), ZScoreNormalization(intensityproperties={})
    with args.routing_csv.open(encoding="utf-8-sig", newline="") as handle:
        route = {row["case_id"]: row["response_state"] for row in csv.DictReader(handle)}
    rows = []
    threshold = float(config["evaluation"]["threshold"])
    for index, case in enumerate(train_cases, 1):
        image, _ = reader.read_images([str(case.image_path)])
        probability = predict_volume(runtime, image, normalizer, config)
        count = int(np.count_nonzero(probability >= threshold))
        rows.append({
            "case_id": case.name,
            "response_state": route[case.name],
            "baseline_predicted_voxels": count,
        })
        print(f"[{index}/{len(train_cases)}] {case.name} foreground={count}", flush=True)
    with (args.output_dir / "train_baseline_foreground.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    no_response = [row for row in rows if row["response_state"] == "No-response"]
    summary = {
        "cases": len(rows),
        "no_response": len(no_response),
        "no_response_all_full_volume_predictions_empty": all(
            row["baseline_predicted_voxels"] == 0 for row in no_response
        ),
        "no_response_nonempty_cases": [
            row["case_id"] for row in no_response if row["baseline_predicted_voxels"] != 0
        ],
    }
    with (args.output_dir / "no_response_sanity.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if not summary["no_response_all_full_volume_predictions_empty"]:
        raise RuntimeError(f"No-response mismatch: {summary['no_response_nonempty_cases']}")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
