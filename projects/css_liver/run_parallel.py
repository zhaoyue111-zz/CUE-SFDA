#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, wait
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]
STAGING = PROJECT / ".staging"
RESULTS = PROJECT / "results"
RECORDS = PROJECT / "experiments.jsonl"
SCRIPT = REPO / "scripts" / "run_cue_sfda.py"
SHARED_CACHE = REPO / "outputs/high_selection/CSS_liver/dynamic_hard_cache"
GT_PATH = Path("/data/hilab/datasets/WCH_MRI145/DWI/liver")
SUCCESS_DICE = 0.75
TRIALS = (
    ("liver_css_anchor_050", 0, PROJECT / "configs/liver_css_anchor_050.yaml"),
    ("liver_css_anchor_075_lrhalf", 1, PROJECT / "configs/liver_css_anchor_075_lrhalf.yaml"),
    ("liver_css_unbalanced_bce", 1, PROJECT / "configs/liver_css_unbalanced_bce.yaml"),
    ("liver_css_consensus_anchor", 1, PROJECT / "configs/liver_css_consensus_anchor.yaml"),
    ("liver_css_consensus_highonly", 1, PROJECT / "configs/liver_css_consensus_highonly.yaml"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validation_curve(artifacts: Path) -> list[dict[str, float | int]]:
    curve = []
    for path in sorted((artifacts / "validation").glob("iteration_*/metrics_summary.csv")):
        iteration = int(path.parent.name.rsplit("_", 1)[1])
        with path.open(encoding="utf-8", newline="") as handle:
            import csv

            row = next(csv.DictReader(handle))
        curve.append({
            "iteration": iteration,
            "dice": float(row["mean_dice"]),
            "precision": float(row["mean_precision"]),
            "recall": float(row["mean_recall"]),
        })
    return curve


def tail(path: Path, count: int = 35) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-count:])


def run_trial(trial: tuple[str, int, Path]) -> dict:
    name, gpu_id, source_config = trial
    started = utc_now()
    work = STAGING / name
    artifacts = work / "artifacts"
    console_log = work / "console.log"
    work.mkdir(parents=True, exist_ok=False)

    with source_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["experiment"]["gpu_id"] = gpu_id
    config["experiment"]["output_dir"] = str(artifacts)
    cache_dir = artifacts / "dynamic_hard_cache"
    cache_metadata = json.loads((SHARED_CACHE / "version_0000/metadata.json").read_text(encoding="utf-8"))
    expected_prompts = config["dynamic_hard"]["prompts"]
    if cache_metadata.get("prompts") != expected_prompts or cache_metadata.get("num_cases") != 101:
        raise RuntimeError("Existing Liver Dynamic-hard cache does not match this experiment")
    artifacts.mkdir(parents=True)
    shutil.copytree(SHARED_CACHE, cache_dir, copy_function=os.link)
    config["dynamic_hard"]["cache_dir"] = str(cache_dir)
    config_path = work / "run_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    env = os.environ.copy()
    src_path = str(REPO / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    python = Path("/home/ubuntu/anaconda3/envs/voxtell/bin/python")
    executable = str(python if python.is_file() else Path(sys.executable))
    command = [executable, str(SCRIPT), "train", "--config", str(config_path), "--gt-path", str(GT_PATH)]

    return_code = -1
    try:
        with console_log.open("w", encoding="utf-8") as output:
            result = subprocess.run(
                command, cwd=REPO, env=env, stdout=output,
                stderr=subprocess.STDOUT, check=False,
            )
            return_code = int(result.returncode)
    except Exception as error:
        exception = f"{type(error).__name__}: {error}"
    else:
        exception = ""

    curve = validation_curve(artifacts)
    best = max(curve, key=lambda item: item["dice"], default=None)
    success = return_code == 0 and best is not None and best["dice"] >= SUCCESS_DICE
    record = {
        "experiment": name,
        "attempt": 1 + sum(
            1 for line in RECORDS.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("experiment") == name
        ) if RECORDS.is_file() else 1,
        "gpu_id": gpu_id,
        "started_at": started,
        "finished_at": utc_now(),
        "status": "success" if success else "failed",
        "return_code": return_code,
        "exception": exception,
        "success_dice_threshold": SUCCESS_DICE,
        "best_validation": best,
        "validation_curve": curve,
        "config": config,
        "settings": {
            "random_walk_score_pooling": config["method"]["random_walk_score_pooling"],
            "random_walk_alpha": config["method"]["random_walk_alpha"],
            "preserve_raw_foreground": config["method"]["preserve_raw_foreground"],
            "high_pseudo_target": config["method"].get("high_pseudo_target", "source"),
            "learning_rate": config["train"]["learning_rate"],
            "high_selection": config["high_selection"],
        },
        "failure_tail": "" if success else tail(console_log),
    }

    if success:
        destination = RESULTS / name
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(artifacts, destination)
        (destination / "run_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        record["result_dir"] = str(destination)
    shutil.rmtree(work, ignore_errors=True)
    with RECORDS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main() -> int:
    STAGING.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        queue = iter(TRIALS)
        active = {}
        for gpu_id in (0, 1):
            trial = next(queue, None)
            if trial is not None:
                name, _, config_path = trial
                future = pool.submit(run_trial, (name, gpu_id, config_path))
                active[future] = gpu_id
        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                gpu_id = active.pop(future)
                result = future.result()
                best = result["best_validation"]
                if best is None:
                    summary = "no validation result"
                else:
                    summary = f"best_dice={best['dice']:.4f}"
                print(
                    f"{result['experiment']} GPU{gpu_id}: {result['status']}, {summary}",
                    flush=True,
                )
                trial = next(queue, None)
                if trial is not None:
                    name, _, config_path = trial
                    next_future = pool.submit(run_trial, (name, gpu_id, config_path))
                    active[next_future] = gpu_id
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
