from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    base = ROOT / "outputs/hierarchical_dynamic"
    for directory in sorted(path for path in base.iterdir() if path.is_dir()):
        summary_path = directory / "training_summary.json"
        checkpoint_path = directory / "checkpoints/latest.pt"
        if not summary_path.is_file() or not checkpoint_path.is_file():
            continue
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("trainer_state", {})
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if "both" in state.get("branch_patch_counts", {}):
            patch_count = int(state["branch_patch_counts"].get("both", 0))
            total = float(state.get("dynamic_hard_loss_total", 0.0))
            summary["dynamic_hard_loss_patches"] = patch_count
            summary["dynamic_hard_loss_mean"] = total / max(patch_count, 1)
        optimizer_steps = state.get("optimizer_step_count")
        if optimizer_steps is None:
            parameter_steps = {
                int(value["step"].item() if hasattr(value["step"], "item") else value["step"])
                for value in checkpoint.get("optimizer_state", {}).get("state", {}).values()
                if "step" in value
            }
            if len(parameter_steps) > 1:
                raise ValueError(
                    f"Inconsistent Adam step counts in {checkpoint_path}: {parameter_steps}"
                )
            optimizer_steps = next(iter(parameter_steps), int(checkpoint.get("iteration", 0)))
        summary["optimizer_steps"] = int(optimizer_steps)
        summary["iterations_completed"] = int(checkpoint.get("iteration", summary.get("iterations_completed", 0)))
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"finalized {directory.name} iterations={summary['iterations_completed']} optimizer_steps={summary['optimizer_steps']}")


if __name__ == "__main__":
    main()
