#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import csv
import fcntl
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
OUT = ROOT / "outputs" / "liver_expansion_control"
RUNS = OUT / "runs"
REPORTS = OUT / "reports"
RECORDS = OUT / "experiment_records.jsonl"
SOURCE_DYNAMIC_CACHE = ROOT / "outputs/high_selection/CSS_liver/dynamic_hard_cache"
GT_PATH = Path("/data/hilab/datasets/WCH_MRI145/DWI/liver")
PYTHON = Path("/home/ubuntu/anaconda3/envs/voxtell/bin/python")
CLI = ROOT / "scripts/run_cue_sfda.py"
TARGET_DICE = 0.75
EXPERIMENTS = ("baseline", "E0", "E1", "E3", "E5", "E2", "E4")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_record(record: dict) -> None:
    RECORDS.parent.mkdir(parents=True, exist_ok=True)
    with RECORDS.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prepare_dynamic_cache(config: dict) -> None:
    source = SOURCE_DYNAMIC_CACHE
    destination = Path(config["dynamic_hard"]["cache_dir"])
    if not source.is_dir():
        raise FileNotFoundError(f"Frozen-source Dynamic-hard cache missing: {source}")
    source_metadata = json.loads(
        (source / "version_0000" / "metadata.json").read_text(encoding="utf-8")
    )
    expected = {
        "num_cases": 101,
        "standard_prompt": "liver",
        "prompts": config["dynamic_hard"]["prompts"],
        "teacher_type": "frozen_source",
    }
    mismatches = {
        key: (source_metadata.get(key), value)
        for key, value in expected.items()
        if source_metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Existing Dynamic-hard cache is incompatible: {mismatches}")
    if destination.exists():
        raise FileExistsError(f"Experiment cache already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, copy_function=os.link)
    # Each experiment starts on the same frozen-source version. Its optional
    # student refresh is isolated under this hard-linked experiment-local root.


def read_validation_curve(run_dir: Path) -> list[dict]:
    curve = []
    for path in sorted((run_dir / "validation").glob("iteration_*/metrics_summary.csv")):
        iteration = int(path.parent.name.rsplit("_", 1)[1])
        with path.open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        curve.append({
            "iteration": iteration,
            **{
                key: float(value)
                for key, value in row.items()
                if key != "num_cases"
            },
        })
    return curve


def copy_failure_report(name: str, config_path: Path, run_dir: Path, console_log: Path) -> None:
    report = REPORTS / name
    report.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, report / "experiment_config.yaml")
    if console_log.is_file():
        shutil.copy2(console_log, report / "console.log")
    for filename in (
        "run.log",
        "resolved_config.json",
        "routing.csv",
        "routing_summary.json",
        "training_summary.json",
        "early_stopping.json",
        "dynamic_hard_cache_metadata.json",
    ):
        source = run_dir / filename
        if source.is_file():
            shutil.copy2(source, report / filename)
    validation_source = run_dir / "validation"
    if validation_source.is_dir():
        for validation_dir in validation_source.glob("iteration_*"):
            destination = report / "validation" / validation_dir.name
            destination.mkdir(parents=True, exist_ok=True)
            for filename in ("metrics_per_case.csv", "metrics_summary.csv"):
                source = validation_dir / filename
                if source.is_file():
                    shutil.copy2(source, destination / filename)


def run_one(name: str, gpu_id: int) -> dict:
    started = now()
    config_path = CONFIG_DIR / f"liver_expansion_control_{name}.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["experiment"]["gpu_id"] = int(gpu_id)
    run_dir = ROOT / config["experiment"]["output_dir"]
    if run_dir.exists():
        raise FileExistsError(f"Experiment output already exists: {run_dir}")
    prepare_dynamic_cache(config)
    console_log = OUT / "console" / f"{name}.log"
    console_log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON), str(CLI), "train",
        "--config", str(config_path),
        "--gt-path", str(GT_PATH),
        "--gpu-id", str(gpu_id),
    ]
    with console_log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    curve = read_validation_curve(run_dir) if run_dir.is_dir() else []
    best = max(curve, key=lambda row: row["mean_dice"], default=None)
    completed = result.returncode == 0 and best is not None
    reached_target = completed and float(best["mean_dice"]) >= TARGET_DICE
    status = "success" if reached_target else "failed"
    if not reached_target:
        copy_failure_report(name, config_path, run_dir, console_log)
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(OUT / "caches" / name, ignore_errors=True)
    record = {
        "experiment": name,
        "gpu_id": int(gpu_id),
        "started_at": started,
        "finished_at": now(),
        "status": status,
        "return_code": int(result.returncode),
        "target_dice": TARGET_DICE,
        "best_validation": best,
        "validation_curve": curve,
        "report_dir": str(REPORTS / name) if not reached_target else str(run_dir),
        "config": str(config_path),
    }
    append_record(record)
    print(
        f"{name}: {status}; best_dice="
        f"{best['mean_dice'] if best else 'n/a'}; report={record['report_dir']}",
        flush=True,
    )
    return record


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(worker, gpu_id, names)
            for gpu_id, names in (
                (0, ("baseline", "E1", "E5", "E4")),
                (1, ("E0", "E3", "E2")),
            )
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def worker(gpu_id: int, names: tuple[str, ...]) -> None:
    for name in names:
        run_one(name, gpu_id)


if __name__ == "__main__":
    main()
