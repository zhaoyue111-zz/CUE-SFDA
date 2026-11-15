from __future__ import annotations

"""Label-free multi-view/multi-prompt teacher consensus.

The cache deliberately stores only the fused maps and confidence maps.  The
individual view predictions are accumulated and released case by case, which
keeps the disk footprint and peak memory bounded.
"""

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from cue_sfda.data import CaseRecord, TargetVolumeStore
from cue_sfda.method.augmentations import StrongViewGenerator
from cue_sfda.modeling.lora import lora_disabled


CACHE_FIELDS = (
    "image_consensus_probability",
    "prompt_consensus_probability",
    "image_voxel_consistency",
    "prompt_voxel_consistency",
    "joint_voxel_confidence",
    "joint_probability",
)


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = max(float(temperature), 1e-6)
    shifted = (values - values.max()) / scale
    weights = np.exp(shifted)
    return (weights / weights.sum()).astype(np.float32)


def _consensus(predictions: np.ndarray, temperature: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fused probability, voxel confidence, and case weights."""
    if predictions.ndim != 5:
        raise ValueError(f"expected [views, 1, D, H, W], got {predictions.shape}")
    values = predictions[:, 0].astype(np.float32, copy=False)
    center = values.mean(axis=0)
    # Reliability is the average agreement with the other views.  The
    # probability range makes 1 - 2*abs error a bounded, interpretable score.
    deviations = np.abs(values - center[None])
    reliability = 1.0 - 2.0 * deviations.reshape(values.shape[0], -1).mean(axis=1)
    reliability = np.clip(reliability, -1.0, 1.0)
    weights = _softmax(reliability, temperature)
    fused = np.tensordot(weights, values, axes=(0, 0)).astype(np.float32)
    confidence = np.clip(1.0 - 2.0 * np.tensordot(weights, deviations, axes=(0, 0)), 0.0, 1.0)
    return fused[None], confidence[None].astype(np.float32), weights


def _geometry_payload(case: CaseRecord, volume: torch.Tensor) -> dict:
    """Create a stable, label-free geometry record for cache validation."""
    path = case.image_path.resolve()
    stat = path.stat()
    payload = {
        "case_id": case.name,
        "image_name": path.name,
        "image_size": int(stat.st_size),
        "image_mtime_ns": int(stat.st_mtime_ns),
        "padded_shape": [int(v) for v in volume.shape[-3:]],
    }
    try:
        import nibabel as nib

        image = nib.load(str(path))
        payload["original_shape"] = [int(v) for v in image.shape]
        payload["affine"] = np.asarray(image.affine, dtype=np.float64).round(6).tolist()
        payload["orientation"] = "|".join(nib.aff2axcodes(image.affine))
    except Exception:
        payload["orientation"] = "unavailable"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signature"] = hashlib.sha256(encoded).hexdigest()
    return payload


class ConsensusCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.cases_dir = self.root / "cases"
        self.metadata_path = self.root / "metadata.json"

    def case_path(self, case_name: str) -> Path:
        stem = Path(case_name).name
        return self.cases_dir / f"{stem}.npz"

    def has_case(self, case_name: str) -> bool:
        return self.case_path(case_name).is_file()

    def load_case(self, case: CaseRecord) -> dict[str, np.ndarray | dict]:
        path = self.case_path(case.name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing consensus cache for {case.name}: {path}")
        with np.load(path, allow_pickle=False) as data:
            arrays = {field: data[field].astype(np.float32, copy=False) for field in CACHE_FIELDS}
            arrays["image_view_case_weights"] = data["image_view_case_weights"].astype(np.float32, copy=False)
            arrays["prompt_case_weights"] = data["prompt_case_weights"].astype(np.float32, copy=False)
            geometry = json.loads(str(data["geometry_json"].item()))
        if geometry.get("case_id") != case.name:
            raise ValueError(f"Consensus cache case mismatch: {path}")
        current_size = int(case.image_path.stat().st_size)
        current_mtime = int(case.image_path.stat().st_mtime_ns)
        if geometry.get("image_size") != current_size or geometry.get("image_mtime_ns") != current_mtime:
            raise ValueError(f"Consensus cache image changed since creation: {case.name}")
        arrays["geometry"] = geometry
        return arrays

    def save_case(
        self,
        case: CaseRecord,
        arrays: dict[str, np.ndarray],
        geometry: dict,
        image_weights: np.ndarray,
        prompt_weights: np.ndarray,
    ) -> None:
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        target = self.case_path(case.name)
        temporary = target.with_suffix(".npz.tmp")
        payload = {field: np.asarray(arrays[field], dtype=np.float32) for field in CACHE_FIELDS}
        payload.update(
            image_view_case_weights=np.asarray(image_weights, dtype=np.float32),
            prompt_case_weights=np.asarray(prompt_weights, dtype=np.float32),
            geometry_json=np.asarray(json.dumps(geometry, sort_keys=True)),
        )
        # np.savez appends .npz when given a string path; a file handle avoids
        # changing the intended temporary name.
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(target)

    def write_metadata(self, payload: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.metadata_path)

    def read_metadata(self) -> dict:
        if not self.metadata_path.is_file():
            return {}
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def patch(self, case: CaseRecord, slicer: tuple[slice, slice, slice]) -> dict[str, torch.Tensor]:
        loaded = self.load_case(case)
        result = {}
        for field in CACHE_FIELDS:
            array = loaded[field]
            result[field] = torch.from_numpy(np.ascontiguousarray(array[(slice(None), *slicer)]))
        return result


class ConsensusCacheBuilder:
    def __init__(
        self,
        config: dict,
        runtime,
        store: TargetVolumeStore,
        views: StrongViewGenerator,
        prompt_embeddings: dict[str, torch.Tensor],
        logger,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.store = store
        self.views = views
        self.prompt_embeddings = prompt_embeddings
        self.logger = logger
        consensus = config.get("consensus", {})
        self.image_temperature = float(consensus.get("image_temperature", 0.15))
        self.prompt_temperature = float(consensus.get("prompt_temperature", 0.15))
        # Refreshes are full-volume teacher inference, not student training.
        # A non-overlapping refresh grid avoids repeating the same expensive
        # forward passes while preserving the stitched full-volume cache.
        self.tile_step_size = float(
            consensus.get(
                "refresh_tile_step_size",
                config["train"].get("tile_step_size", 0.5),
            )
        )
        self.amp_enabled = bool(config["model"].get("amp", True)) and runtime.device.type == "cuda"
        self.disable_lora = True

    def _autocast(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def build_case(self, case: CaseRecord, cache: ConsensusCache, cycle: int) -> None:
        self._case = case
        volume, _ = self.store.sliding_window(case, self.tile_step_size)
        geometry = _geometry_payload(case, volume)
        geometry["cache_model_version"] = int(cycle)
        geometry["refresh_cycle"] = int(cycle)
        was_training = self.runtime.model.training
        self.runtime.model.eval()
        lora_context = lora_disabled(self.runtime.model) if self.disable_lora else nullcontext()
        with torch.no_grad(), lora_context:
            image_predictions = []
            for view_index in range(int(self.config["method"]["num_views"])):
                per_patch = []
                for patch_index, slicer in enumerate(self._slicers):
                    image = self.store.extract_patch(volume, slicer).unsqueeze(0).to(self.runtime.device)
                    # Recreate the deterministic view sequence one tile at a
                    # time.  The view generator has no spatial transform, so
                    # alignment is identity and the same definition as Stage 1
                    # is used without retaining intermediate maps.
                    generated = self.views.make_uncertainty_views(image)
                    view = generated[view_index]
                    with self._autocast():
                        logits = self.runtime.model(view.image, self.prompt_embeddings[str(self.config["consensus"].get("standard_prompt", self.config["model"]["prompt"]))])
                    per_patch.append(torch.sigmoid(logits.float())[0, 0].cpu().numpy())
                image_predictions.append(self._stitch_numpy(per_patch, volume.shape[-3:], self._slicers))
            prompt_predictions = []
            raw_prompt = self.prompt_embeddings
            for prompt in self.config["consensus"]["prompts"]:
                per_patch = []
                for slicer in self._slicers:
                    image = self.store.extract_patch(volume, slicer).unsqueeze(0).to(self.runtime.device)
                    with self._autocast():
                        logits = self.runtime.model(image, raw_prompt[str(prompt)])
                    per_patch.append(torch.sigmoid(logits.float())[0, 0].cpu().numpy())
                prompt_predictions.append(self._stitch_numpy(per_patch, volume.shape[-3:], self._slicers))
        self.runtime.model.train(was_training)
        image_tensor = np.stack(image_predictions, axis=0)[:, None]
        prompt_tensor = np.stack(prompt_predictions, axis=0)[:, None]
        image_probability, image_confidence, image_weights = _consensus(image_tensor, self.image_temperature)
        prompt_probability, prompt_confidence, prompt_weights = _consensus(prompt_tensor, self.prompt_temperature)
        joint_confidence = np.sqrt(np.clip(image_confidence * prompt_confidence, 0.0, 1.0)).astype(np.float32)
        joint_probability = (0.5 * image_probability + 0.5 * prompt_probability).astype(np.float32)
        cache.save_case(
            case,
            {
                "image_consensus_probability": image_probability,
                "prompt_consensus_probability": prompt_probability,
                "image_voxel_consistency": image_confidence,
                "prompt_voxel_consistency": prompt_confidence,
                "joint_voxel_confidence": joint_confidence,
                "joint_probability": joint_probability,
            },
            geometry,
            image_weights,
            prompt_weights,
        )
        self.logger.info("consensus_cache cycle=%d case=%s views=%d prompts=%d shape=%s", cycle, case.name, len(image_predictions), len(prompt_predictions), tuple(volume.shape[-3:]))

    @staticmethod
    def _stitch_numpy(patches: list[np.ndarray], shape: tuple[int, ...], slicers: list[tuple[slice, slice, slice]]) -> np.ndarray:
        result = np.zeros(shape, dtype=np.float32)
        counts = np.zeros(shape, dtype=np.float32)
        for patch, slicer in zip(patches, slicers):
            result[slicer] += patch
            counts[slicer] += 1.0
        return result / np.maximum(counts, 1.0)

    def build(
        self,
        cases: list[CaseRecord],
        cache: ConsensusCache,
        cycle: int,
        metadata_extra: dict | None = None,
    ) -> None:
        self.logger.info("building consensus cache: cycle=%d cases=%d", cycle, len(cases))
        for case in cases:
            volume, slicers = self.store.sliding_window(case, self.tile_step_size)
            self._slicers = slicers
            self.build_case(case, cache, cycle)
        metadata = {
            "cache_version": int(cycle),
            "num_cases": len(cases),
            "case_ids": [case.name for case in cases],
            "num_views": int(self.config["method"]["num_views"]),
            "prompts": [str(v) for v in self.config["consensus"]["prompts"]],
            "standard_prompt": str(self.config["consensus"].get("standard_prompt", self.config["model"]["prompt"])),
            "teacher": "frozen_snapshot",
        }
        if metadata_extra:
            metadata.update(metadata_extra)
        cache.write_metadata(metadata)
