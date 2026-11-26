from __future__ import annotations

import fcntl
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from nnunetv2.inference.sliding_window_prediction import compute_gaussian

from cue_sfda.data import CaseRecord, TargetVolumeStore
from cue_sfda.method.dynamic_hard import source_checkpoint_signature
from cue_sfda.modeling.lora import lora_disabled


@dataclass(frozen=True)
class SourceVolumePrediction:
    probability: np.ndarray
    spacing: tuple[float, float, float]
    padded_shape: tuple[int, int, int]
    roi_slicer: tuple[slice, slice, slice]

    @property
    def soft_volume(self) -> float:
        return float(self.probability.astype(np.float64).sum())

    @property
    def foreground_voxels(self) -> int:
        return int((self.probability >= 0.5).sum())

    def padded_probability(self, device: torch.device) -> torch.Tensor:
        output = torch.zeros(self.padded_shape, dtype=torch.float32, device=device)
        output[self.roi_slicer] = torch.as_tensor(
            np.asarray(self.probability, dtype=np.float32), device=device
        )
        return output[None, None]

    def padded_mask(self, mask: np.ndarray, device: torch.device) -> torch.Tensor:
        output = torch.zeros(self.padded_shape, dtype=torch.bool, device=device)
        output[self.roi_slicer] = torch.as_tensor(mask, dtype=torch.bool, device=device)
        return output[None, None]


class SourceVolumeCache:
    """Locked, geometry-checked cache of frozen-source full-case probabilities."""

    def __init__(self, root: str | Path, prompt: str, model_config: dict) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.prompt = str(prompt)
        self.checkpoint = source_checkpoint_signature(model_config)

    @staticmethod
    def _slice_pairs(slicers: tuple[slice, slice, slice]) -> list[list[int]]:
        return [[int(s.start or 0), int(s.stop)] for s in slicers]

    def _paths(self, case: CaseRecord) -> tuple[Path, Path]:
        identity = json.dumps(
            {"prompt": self.prompt, "checkpoint": self.checkpoint},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        path = self.root / digest / f"{case.name}.npz"
        return path, path.with_suffix(".lock")

    def _metadata(
        self,
        case: CaseRecord,
        geometry: dict[str, object],
        tile_step_size: float,
        patch_size: tuple[int, int, int],
    ) -> dict:
        stat = case.image_path.stat()
        return {
            "schema_version": 1,
            "case_id": case.name,
            "image_size": int(stat.st_size),
            "image_mtime_ns": int(stat.st_mtime_ns),
            "prompt": self.prompt,
            "source_checkpoint": self.checkpoint,
            "patch_size": [int(value) for value in patch_size],
            "tile_step_size": float(tile_step_size),
            "roi_shape": [int(value) for value in geometry["roi_shape"]],
            "padded_shape": [int(value) for value in geometry["padded_shape"]],
            "roi_slicer": self._slice_pairs(geometry["roi_slicer"]),
            "spacing": [float(value) for value in geometry["spacing"]],
        }

    def get_or_build(
        self,
        case: CaseRecord,
        volume: torch.Tensor,
        slicers: list[tuple[slice, slice, slice]],
        geometry: dict[str, object],
        store: TargetVolumeStore,
        runtime,
        embedding: torch.Tensor,
        model_config: dict,
        tile_step_size: float,
        logger: logging.Logger | None = None,
    ) -> SourceVolumePrediction:
        path, lock_path = self._paths(case)
        expected = self._metadata(case, geometry, tile_step_size, tuple(runtime.patch_size))
        path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                result = self._read(path, expected)
                if result is not None:
                    return result
                prediction = self._predict(
                    case,
                    volume,
                    slicers,
                    geometry,
                    store,
                    runtime,
                    embedding,
                    model_config,
                )
                roi = prediction.probability.astype(np.float32, copy=False)
                temporary = path.with_name(f"{path.name}.{__import__('os').getpid()}.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        probability=roi,
                        metadata=np.asarray(json.dumps(expected, sort_keys=True)),
                    )
                temporary.replace(path)
                if logger is not None:
                    logger.info(
                        "source_volume_cache built case=%s prompt=%s roi=%s padded=%s spacing=%s soft_volume=%.1f",
                        case.name,
                        self.prompt,
                        expected["roi_shape"],
                        expected["padded_shape"],
                        expected["spacing"],
                        prediction.soft_volume,
                    )
                return prediction
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _read(path: Path, expected: dict) -> SourceVolumePrediction | None:
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata"].item()))
                if metadata != expected:
                    return None
                probability = payload["probability"].astype(np.float32)
                if probability.shape != tuple(expected["roi_shape"]):
                    return None
            roi_slicer = tuple(
                slice(int(start), int(stop)) for start, stop in metadata["roi_slicer"]
            )
            return SourceVolumePrediction(
                probability=probability,
                spacing=tuple(float(value) for value in metadata["spacing"]),
                padded_shape=tuple(int(value) for value in metadata["padded_shape"]),
                roi_slicer=roi_slicer,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    @staticmethod
    def _predict(
        case: CaseRecord,
        volume: torch.Tensor,
        slicers: list[tuple[slice, slice, slice]],
        geometry: dict[str, object],
        store: TargetVolumeStore,
        runtime,
        embedding: torch.Tensor,
        model_config: dict,
    ) -> SourceVolumePrediction:
        device = runtime.device
        padded_shape = tuple(int(value) for value in geometry["padded_shape"])
        patch_size = tuple(int(value) for value in runtime.patch_size)
        amp_enabled = bool(model_config.get("amp", True)) and device.type == "cuda"
        accumulation = torch.zeros(padded_shape, dtype=torch.float32, device=device)
        normalization = torch.zeros_like(accumulation)
        gaussian = compute_gaussian(
            patch_size,
            sigma_scale=1.0 / 8.0,
            value_scaling_factor=10,
            dtype=torch.float32,
            device=device,
        )
        was_training = runtime.model.training
        runtime.model.eval()
        try:
            with torch.inference_mode(), lora_disabled(runtime.model):
                for slicer in slicers:
                    image = store.extract_patch(volume, slicer).unsqueeze(0)
                    image = image.to(device, non_blocking=True)
                    if amp_enabled:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            logits = runtime.model(image, embedding)
                    else:
                        logits = runtime.model(image, embedding)
                    accumulation[slicer] += logits[0, 0].float() * gaussian
                    normalization[slicer] += gaussian
            fused_probability = torch.sigmoid(
                accumulation / normalization.clamp_min(1e-6)
            )
            roi_slicer = geometry["roi_slicer"]
            probability = fused_probability[roi_slicer].cpu().numpy().astype(
                np.float32
            )
        finally:
            runtime.model.train(was_training)
        return SourceVolumePrediction(
            probability=probability,
            spacing=tuple(float(value) for value in geometry["spacing"]),
            padded_shape=padded_shape,
            roi_slicer=roi_slicer,
        )
