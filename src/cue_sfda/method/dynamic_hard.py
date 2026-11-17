from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from cue_sfda.data import CaseRecord, TargetVolumeStore
from cue_sfda.method.augmentations import StrongViewGenerator
from cue_sfda.modeling.lora import lora_disabled


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scaled = max(float(temperature), 1e-6)
    weights = np.exp((values - values.max()) / scaled)
    return (weights / weights.sum()).astype(np.float32)


def consensus(
    predictions: np.ndarray, temperature: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse views and compute their agreement, matching historical Dynamic-hard."""
    if predictions.ndim != 5:
        raise ValueError(
            f"expected predictions [views, 1, D, H, W], got {predictions.shape}"
        )
    values = predictions[:, 0].astype(np.float32, copy=False)
    center = values.mean(axis=0)
    deviations = np.abs(values - center[None])
    reliability = 1.0 - 2.0 * deviations.reshape(values.shape[0], -1).mean(axis=1)
    reliability = np.clip(reliability, -1.0, 1.0)
    weights = _softmax(reliability, temperature)
    fused = np.tensordot(weights, values, axes=(0, 0)).astype(np.float32)
    confidence = np.clip(
        1.0 - 2.0 * np.tensordot(weights, deviations, axes=(0, 0)), 0.0, 1.0
    )
    return fused[None], confidence[None].astype(np.float32), weights


def refresh_ready(
    seen_cases: set[str], case_names: set[str], refresh_cycle: int, max_cycles: int
) -> bool:
    return refresh_cycle < max_cycles and seen_cases == case_names


class DynamicHardCache:
    """Versioned, experiment-local cache for joint Dynamic-hard targets."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.metadata_path = self.root / "metadata.json"

    def version_dir(self, version: int) -> Path:
        return self.root / f"version_{int(version):04d}" / "cases"

    def case_path(self, case_name: str, version: int) -> Path:
        return self.version_dir(version) / f"{Path(case_name).name}.npz"

    def has_version(self, version: int, cases: list[CaseRecord]) -> bool:
        return all(self.case_path(case.name, version).is_file() for case in cases)

    def load_case(
        self, case: CaseRecord, version: int
    ) -> dict[str, np.ndarray | dict]:
        path = self.case_path(case.name, version)
        with np.load(path, allow_pickle=False) as payload:
            probability = payload["joint_probability"].astype(np.float32, copy=False)
            confidence = payload["joint_voxel_confidence"].astype(
                np.float32, copy=False
            )
            geometry = json.loads(str(payload["geometry_json"].item()))
        stat = case.image_path.stat()
        if geometry.get("case_id") != case.name:
            raise ValueError(f"Dynamic-hard cache case mismatch: {path}")
        if (
            geometry.get("image_size") != int(stat.st_size)
            or geometry.get("image_mtime_ns") != int(stat.st_mtime_ns)
        ):
            raise ValueError(f"Dynamic-hard cache image changed: {case.name}")
        return {
            "joint_probability": probability,
            "joint_voxel_confidence": confidence,
            "geometry": geometry,
        }

    def save_case(
        self,
        case: CaseRecord,
        version: int,
        probability: np.ndarray,
        confidence: np.ndarray,
        geometry: dict,
    ) -> None:
        target = self.case_path(case.name, version)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                joint_probability=np.asarray(probability, dtype=np.float32),
                joint_voxel_confidence=np.asarray(confidence, dtype=np.float32),
                geometry_json=np.asarray(json.dumps(geometry, sort_keys=True)),
            )
        temporary.replace(target)

    def metadata(self) -> dict:
        if not self.metadata_path.is_file():
            return {}
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def activate_version(self, payload: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.metadata_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.metadata_path)


class DynamicHardCacheBuilder:
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
        dynamic_config = config["dynamic_hard"]
        self.image_temperature = float(dynamic_config["image_temperature"])
        self.prompt_temperature = float(dynamic_config["prompt_temperature"])
        self.tile_step_size = float(dynamic_config["refresh_tile_step_size"])
        self.amp_enabled = bool(config["model"].get("amp", True)) and (
            runtime.device.type == "cuda"
        )
        self.disable_lora = True

    def _autocast(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    @staticmethod
    def _stitch(
        patches: list[np.ndarray],
        shape: tuple[int, ...],
        slicers: list[tuple[slice, slice, slice]],
    ) -> np.ndarray:
        result = np.zeros(shape, dtype=np.float32)
        counts = np.zeros(shape, dtype=np.float32)
        for patch, slicer in zip(patches, slicers, strict=True):
            result[slicer] += patch
            counts[slicer] += 1.0
        return result / np.maximum(counts, 1.0)

    def build_case(
        self,
        case: CaseRecord,
        cache: DynamicHardCache,
        version: int,
        volume: torch.Tensor,
        slicers: list[tuple[slice, slice, slice]],
    ) -> None:
        image_predictions = []
        was_training = self.runtime.model.training
        self.runtime.model.eval()
        lora_context = (
            lora_disabled(self.runtime.model) if self.disable_lora else nullcontext()
        )
        with torch.no_grad(), lora_context:
            for view_index in range(int(self.config["method"]["num_views"])):
                patches = []
                for slicer in slicers:
                    image = self.store.extract_patch(volume, slicer).unsqueeze(0)
                    image = image.to(self.runtime.device)
                    # Match the historical view generator call order exactly.
                    view = self.views.make_uncertainty_views(image)[view_index]
                    with self._autocast():
                        logits = self.runtime.model(
                            view.image,
                            self.prompt_embeddings[
                                str(self.config["dynamic_hard"]["standard_prompt"])
                            ],
                        )
                    patches.append(torch.sigmoid(logits.float())[0, 0].cpu().numpy())
                image_predictions.append(
                    self._stitch(patches, tuple(volume.shape[-3:]), slicers)
                )

            prompt_predictions = []
            for prompt in self.config["dynamic_hard"]["prompts"]:
                patches = []
                for slicer in slicers:
                    image = self.store.extract_patch(volume, slicer).unsqueeze(0)
                    image = image.to(self.runtime.device)
                    with self._autocast():
                        logits = self.runtime.model(
                            image, self.prompt_embeddings[str(prompt)]
                        )
                    patches.append(torch.sigmoid(logits.float())[0, 0].cpu().numpy())
                prompt_predictions.append(
                    self._stitch(patches, tuple(volume.shape[-3:]), slicers)
                )
        self.runtime.model.train(was_training)

        image_probability, image_confidence, _ = consensus(
            np.stack(image_predictions, axis=0)[:, None], self.image_temperature
        )
        prompt_probability, prompt_confidence, _ = consensus(
            np.stack(prompt_predictions, axis=0)[:, None], self.prompt_temperature
        )
        joint_probability = (0.5 * image_probability + 0.5 * prompt_probability).astype(
            np.float32
        )
        joint_confidence = np.sqrt(
            np.clip(image_confidence * prompt_confidence, 0.0, 1.0)
        ).astype(np.float32)
        stat = case.image_path.stat()
        geometry = {
            "case_id": case.name,
            "image_size": int(stat.st_size),
            "image_mtime_ns": int(stat.st_mtime_ns),
            "padded_shape": [int(value) for value in volume.shape[-3:]],
            "cache_version": int(version),
        }
        cache.save_case(
            case, version, joint_probability, joint_confidence, geometry
        )
        self.logger.info(
            "dynamic_hard_cache version=%d case=%s views=%d prompts=%d shape=%s",
            version,
            case.name,
            len(image_predictions),
            len(prompt_predictions),
            tuple(volume.shape[-3:]),
        )

    def build(
        self,
        cases: list[CaseRecord],
        cache: DynamicHardCache,
        version: int,
        teacher_type: str,
    ) -> None:
        self.logger.info(
            "building Dynamic-hard cache version=%d cases=%d teacher=%s",
            version,
            len(cases),
            teacher_type,
        )
        was_training = self.runtime.model.training
        try:
            for case in cases:
                volume, slicers = self.store.sliding_window(
                    case, self.tile_step_size
                )
                self.build_case(case, cache, version, volume, slicers)
        finally:
            self.runtime.model.train(was_training)
        cache.activate_version(
            {
                "cache_version": int(version),
                "active_version": int(version),
                "teacher_type": teacher_type,
                "num_cases": len(cases),
                "case_ids": [case.name for case in cases],
                "num_views": int(self.config["method"]["num_views"]),
                "prompts": [str(prompt) for prompt in self.config["dynamic_hard"]["prompts"]],
                "standard_prompt": str(
                    self.config["dynamic_hard"]["standard_prompt"]
                ),
            }
        )
