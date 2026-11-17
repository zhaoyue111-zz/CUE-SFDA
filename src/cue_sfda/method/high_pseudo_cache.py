from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cue_sfda.data import CaseRecord, TargetVolumeStore
from cue_sfda.method.refinement import affinity_random_walk
from cue_sfda.modeling.lora import lora_disabled


class HighPseudoCache:
    """Shared frozen-teacher pseudo labels and foreground D5 case prototypes."""

    def __init__(self, root: str | Path, expected_cases: list[str] | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"High pseudo-label cache is incomplete: {self.root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not self.metadata.get("complete"):
            raise ValueError(f"High pseudo-label cache is marked incomplete: {self.root}")
        if expected_cases is not None and not set(expected_cases).issubset(set(self.metadata["case_ids"])):
            raise ValueError("High pseudo-label cache does not cover all requested cases")
        self._targets: dict[str, np.ndarray] = {}
        self._counts: dict[str, np.ndarray] = {}
        self._prototype_path = self.root / "case_prototypes.pt"
        self._prototypes: dict[str, torch.Tensor] | None = None

    @property
    def prototypes(self) -> dict[str, torch.Tensor]:
        """Load prototypes only for explicitly enabled relation objectives."""
        if self._prototypes is None:
            self._prototypes = torch.load(
                self._prototype_path, map_location="cpu", weights_only=True
            )
        return self._prototypes

    def _load_case(self, case_name: str) -> None:
        if case_name in self._targets:
            return
        stem = Path(case_name).stem
        self._targets[case_name] = np.load(
            self.root / f"{stem}.targets.npy", mmap_mode="r"
        )
        self._counts[case_name] = np.load(
            self.root / f"{stem}.counts.npy", mmap_mode="r"
        )

    def target(self, case_name: str, patch_index: int, device: torch.device) -> tuple[torch.Tensor, int, int]:
        self._load_case(case_name)
        target = torch.from_numpy(np.array(self._targets[case_name][patch_index], copy=True))
        counts = self._counts[case_name][patch_index]
        return target.to(device=device, dtype=torch.float32)[None, None], int(counts[0]), int(counts[1])

    def first_effective_patch(self, case_name: str, minimum_foreground_voxels: int) -> int | None:
        self._load_case(case_name)
        values = np.asarray(self._counts[case_name][:, 1])
        matches = np.flatnonzero(values >= int(minimum_foreground_voxels))
        return int(matches[0]) if matches.size else None


def _source_signature(config: dict, case_ids: list[str]) -> dict:
    checkpoint = Path(config["model"]["model_dir"]).expanduser().resolve() / "checkpoint_final.pth"
    stat = checkpoint.stat()
    method = config["method"]
    return {
        "case_ids": case_ids,
        "prompt": str(config["model"]["prompt"]),
        "patch_size": [int(v) for v in config["data"]["patch_size"]],
        "threshold": float(method["probability_threshold"]),
        "random_walk_size": [int(v) for v in method["random_walk_size"]],
        "random_walk_steps": int(method["random_walk_steps"]),
        "random_walk_alpha": float(method["random_walk_alpha"]),
        "affinity_temperature": float(method["affinity_temperature"]),
        "random_walk_score_pooling": str(method.get("random_walk_score_pooling", "average")),
        "preserve_raw_foreground": bool(method.get("preserve_raw_foreground", False)),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_size": int(stat.st_size),
        "source_checkpoint_mtime_ns": int(stat.st_mtime_ns),
        "target_dtype": "float16",
    }


def build_initial_high_pseudo_cache(
    config: dict,
    runtime,
    cases: list[CaseRecord],
    cache_dir: str | Path,
    logger,
) -> HighPseudoCache:
    """Build deterministic source targets once; no GT enters this function."""
    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    case_ids = sorted(case.name for case in cases)
    signature = _source_signature(config, case_ids)
    metadata_path = root / "metadata.json"
    if metadata_path.is_file():
        current = json.loads(metadata_path.read_text(encoding="utf-8"))
        if current.get("complete") and all(current.get(k) == v for k, v in signature.items()):
            try:
                return HighPseudoCache(root, expected_cases=case_ids)
            except (OSError, ValueError, KeyError):
                pass

    patch_size = tuple(int(v) for v in config["data"]["patch_size"])
    store = TargetVolumeStore(patch_size)
    method = config["method"]
    threshold = float(method["probability_threshold"])
    graph_size = tuple(int(v) for v in method["random_walk_size"])
    amp_enabled = bool(config["model"].get("amp", True)) and runtime.device.type == "cuda"
    prototypes: dict[str, torch.Tensor] = {}
    was_training = runtime.model.training
    runtime.model.eval()
    try:
        with torch.inference_mode():
            for case_number, case in enumerate(sorted(cases, key=lambda c: c.name), start=1):
                volume, slicers = store.sliding_window(case, float(config["train"].get("tile_step_size", 0.5)))
                if not slicers:
                    raise RuntimeError(f"No training patches generated for {case.name}")
                stem = Path(case.name).stem
                temp_target = root / f".{stem}.targets.tmp.npy"
                temp_counts = root / f".{stem}.counts.tmp.npy"
                target_array = np.lib.format.open_memmap(
                    temp_target, mode="w+", dtype=np.float16,
                    shape=(len(slicers), *patch_size),
                )
                count_array = np.lib.format.open_memmap(
                    temp_counts, mode="w+", dtype=np.int32, shape=(len(slicers), 2)
                )
                feature_sum: torch.Tensor | None = None
                feature_count = 0.0
                for patch_index, slicer in enumerate(slicers):
                    patch = store.extract_patch(volume, slicer).unsqueeze(0).to(runtime.device)
                    autocast = torch.autocast("cuda", dtype=torch.float16) if amp_enabled else torch.autocast("cpu", enabled=False)
                    with autocast, lora_disabled(runtime.model):
                        logits, features = runtime.model(patch, runtime.embedding, return_features=True)
                    probability = torch.sigmoid(logits.float())
                    raw_count = int((probability >= threshold).sum().item())
                    refined = affinity_random_walk(
                        probability=probability,
                        features=features,
                        image=patch,
                        output_size=tuple(probability.shape[-3:]),
                        graph_size=graph_size,
                        steps=int(method["random_walk_steps"]),
                        alpha=float(method["random_walk_alpha"]),
                        temperature=float(method["affinity_temperature"]),
                        score_pooling=str(method.get("random_walk_score_pooling", "average")),
                    ).clamp(0.0, 1.0)
                    if bool(method.get("preserve_raw_foreground", False)):
                        refined = torch.maximum(refined, probability)
                    hard_mask = (refined >= threshold).float()
                    refined_count = int(hard_mask.sum().item())
                    target_array[patch_index] = refined[0, 0].to(torch.float16).cpu().numpy()
                    count_array[patch_index] = (raw_count, refined_count)

                    feature_mask = hard_mask
                    if tuple(feature_mask.shape[-3:]) != tuple(features.shape[-3:]):
                        feature_mask = F.adaptive_max_pool3d(feature_mask, features.shape[-3:])
                    pooled_count = float(feature_mask.sum().item())
                    if pooled_count > 0:
                        patch_sum = (features * feature_mask.to(features.dtype)).sum(
                            dim=(0, 2, 3, 4), dtype=torch.float32
                        )
                        if feature_sum is None:
                            feature_sum = torch.zeros_like(patch_sum)
                        feature_sum += patch_sum
                        feature_count += pooled_count
                    del patch, logits, features, probability, refined, hard_mask, feature_mask
                target_array.flush()
                count_array.flush()
                del target_array, count_array
                (root / f"{stem}.targets.npy").unlink(missing_ok=True)
                (root / f"{stem}.counts.npy").unlink(missing_ok=True)
                temp_target.replace(root / f"{stem}.targets.npy")
                temp_counts.replace(root / f"{stem}.counts.npy")
                if feature_sum is not None and feature_count > 0:
                    prototype = F.normalize(feature_sum / feature_count, dim=0, eps=1e-6).cpu()
                    prototypes[case.name] = prototype
                logger.info(
                    "frozen_high_cache=%d/%d case=%s patches=%d fg_prototype=%s",
                    case_number, len(cases), case.name, len(slicers), case.name in prototypes,
                )
                del volume
                if runtime.device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        runtime.model.train(was_training)

    torch.save(prototypes, root / "case_prototypes.pt")
    metadata = {
        **signature,
        "complete": True,
        "cached_cases": len(cases),
        "prototype_cases": sorted(prototypes),
        "prototype_count": len(prototypes),
        "teacher": "frozen_source_voxtell_lora_disabled",
        "gt_used": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Frozen high pseudo cache complete: cases=%d prototypes=%d path=%s", len(cases), len(prototypes), root)
    return HighPseudoCache(root, expected_cases=case_ids)
