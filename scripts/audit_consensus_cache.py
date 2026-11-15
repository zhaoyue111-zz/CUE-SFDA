#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cue_sfda.config import load_config
from cue_sfda.data import load_split
from cue_sfda.method.consensus import CACHE_FIELDS, ConsensusCache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    train, test = load_split(config["data"]["image_dir"], config["data"]["split_json"])
    cache = ConsensusCache(args.cache_dir)
    expected = {case.name for case in train}
    metadata = cache.read_metadata()
    cached = set(metadata.get("case_ids", []))
    errors = []
    if cached != expected:
        errors.append({"type": "case_set", "missing": sorted(expected - cached), "extra": sorted(cached - expected)})
    if cached.intersection({case.name for case in test}):
        errors.append({"type": "test_case_in_cache"})
    checked = 0
    for case in train:
        try:
            data = cache.load_case(case)
            shape = tuple(int(v) for v in data["geometry"]["padded_shape"])
            for field in CACHE_FIELDS:
                if data[field].shape != (1, *shape):
                    errors.append({"type": "shape", "case": case.name, "field": field, "shape": list(data[field].shape), "expected": [1, *shape]})
                if not np.isfinite(data[field]).all():
                    errors.append({"type": "nonfinite", "case": case.name, "field": field})
            checked += 1
        except Exception as exc:
            errors.append({"type": "load", "case": case.name, "error": str(exc)})
    result = {"ok": not errors, "train_cases": len(train), "test_cases": len(test), "checked_cases": checked, "metadata": metadata, "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

