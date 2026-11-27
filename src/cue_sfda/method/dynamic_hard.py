from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from nnunetv2.inference.sliding_window_prediction import compute_gaussian

from cue_sfda.benchmark.metrics import pairwise_hard
from cue_sfda.data import CaseRecord, TargetVolumeStore
from cue_sfda.method.augmentations import StrongViewGenerator
from cue_sfda.method.losses import masked_binary_cross_entropy
from cue_sfda.modeling.lora import lora_disabled


def source_checkpoint_signature(model_config: dict) -> dict | None:
    model_dir = model_config.get("model_dir")
    if not model_dir:
        return None
    path = Path(model_dir).expanduser().resolve() / "checkpoint_final.pth"
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


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


def sample_coverage_index(
    high_indices: list[int],
    low_indices: list[int],
    case_names: list[str],
    seen_cases: set[str],
    high_probability: float,
    generator: torch.Generator,
    coverage_mode: bool = False,
) -> int:
    """Use ordinary stratified sampling or resume-only uniform coverage completion."""
    if coverage_mode:
        unseen = [
            index
            for index in high_indices + low_indices
            if case_names[index] not in seen_cases
        ]
        if unseen:
            selected = int(torch.randint(len(unseen), (1,), generator=generator).item())
            return unseen[selected]
    use_high = bool(torch.rand((), generator=generator) < high_probability)
    selected_branch = high_indices if use_high else low_indices
    other_branch = low_indices if use_high else high_indices
    candidates = selected_branch or other_branch
    selected = int(torch.randint(len(candidates), (1,), generator=generator).item())
    return candidates[selected]


def dynamic_hard_consensus_loss(
    student_probability: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    confidence_threshold: float = 0.0,
) -> torch.Tensor:
    """Historical Low-branch weighted BCE used by Dynamic-hard."""
    weight = confidence
    if confidence_threshold > 0.0:
        weight = weight * (weight >= confidence_threshold).float()
    return masked_binary_cross_entropy(
        student_probability,
        target.detach(),
        weight.detach().clamp_min(1e-3),
    )


class DynamicHardCache:
    """Versioned, experiment-local cache for joint Dynamic-hard targets."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.metadata_path = self.root / "metadata.json"

    def version_dir(self, version: int) -> Path:
        return self.root / f"version_{int(version):04d}"

    def version_cases_dir(self, version: int) -> Path:
        return self.version_dir(version) / "cases"

    def version_metadata_path(self, version: int) -> Path:
        return self.version_dir(version) / "metadata.json"

    def case_path(self, case_name: str, version: int) -> Path:
        return self.version_cases_dir(version) / f"{Path(case_name).name}.npz"

    def has_version(self, version: int, cases: list[CaseRecord]) -> bool:
        return all(self.case_path(case.name, version).is_file() for case in cases)

    def case_confidences(self, version: int) -> dict[str, float]:
        path = self.version_dir(version) / "case_confidences.json"
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = {str(case): float(value) for case, value in payload.items()}
        if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in result.values()):
            raise ValueError(f"Invalid case confidence values in {path}")
        return result

    def load_case(
        self, case: CaseRecord, version: int
    ) -> dict[str, np.ndarray | dict]:
        path = self.case_path(case.name, version)
        with np.load(path, allow_pickle=False) as payload:
            probability = payload["joint_probability"].astype(np.float32, copy=False)
            confidence = payload["joint_voxel_confidence"].astype(
                np.float32, copy=False
            )
            pixel_weight = (
                payload["pixel_weight"].astype(np.float32, copy=False)
                if "pixel_weight" in payload.files
                else None
            )
            quality_prediction_mask = (
                payload["quality_prediction_mask"].astype(np.uint8, copy=False)
                if "quality_prediction_mask" in payload.files
                else None
            )
            geometry = json.loads(str(payload["geometry_json"].item()))
        if (
            probability.ndim != 4
            or probability.shape[0] != 1
            or confidence.shape != probability.shape
            or geometry.get("padded_shape") != [int(v) for v in probability.shape[-3:]]
            or (pixel_weight is not None and pixel_weight.shape != probability.shape)
            or (
                quality_prediction_mask is not None
                and quality_prediction_mask.shape != probability.shape
            )
        ):
            raise ValueError(f"Dynamic-hard cache array geometry is invalid: {path}")
        if (
            not np.isfinite(probability).all()
            or not np.isfinite(confidence).all()
            or np.any(probability < 0.0)
            or np.any(probability > 1.0)
            or np.any(confidence < 0.0)
            or np.any(confidence > 1.0)
            or (
                quality_prediction_mask is not None
                and np.any((quality_prediction_mask != 0) & (quality_prediction_mask != 1))
            )
            or (
                pixel_weight is not None
                and (
                    not np.isfinite(pixel_weight).all()
                    or np.any(pixel_weight < 0.0)
                    or np.any(pixel_weight > 1.0)
                )
            )
        ):
            raise ValueError(f"Dynamic-hard cache contains invalid values: {path}")
        stat = case.image_path.stat()
        if geometry.get("case_id") != case.name:
            raise ValueError(f"Dynamic-hard cache case mismatch: {path}")
        version = int(path.parent.parent.name.split("_")[-1])
        version_metadata = self.metadata(version)
        if (
            int(geometry.get("cache_version", -1)) != version
            or geometry.get("teacher_type") != version_metadata.get("teacher_type")
            or geometry.get("teacher_iteration")
            != version_metadata.get("teacher_iteration")
        ):
            raise ValueError(f"Dynamic-hard cache version/type mismatch: {path}")
        if (
            geometry.get("image_size") != int(stat.st_size)
            or geometry.get("image_mtime_ns") != int(stat.st_mtime_ns)
        ):
            raise ValueError(f"Dynamic-hard cache image changed: {case.name}")
        result = {
            "joint_probability": probability,
            "joint_voxel_confidence": confidence,
            "geometry": geometry,
        }
        if pixel_weight is not None:
            result["pixel_weight"] = pixel_weight
        if quality_prediction_mask is not None:
            result["quality_prediction_mask"] = quality_prediction_mask
        return result

    def save_case(
        self,
        case: CaseRecord,
        version: int,
        probability: np.ndarray,
        confidence: np.ndarray,
        geometry: dict,
        pixel_weight: np.ndarray | None = None,
        quality_prediction_mask: np.ndarray | None = None,
    ) -> None:
        target = self.case_path(case.name, version)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            arrays = {
                "joint_probability": np.asarray(probability, dtype=np.float32),
                "joint_voxel_confidence": np.asarray(confidence, dtype=np.float32),
                "geometry_json": np.asarray(json.dumps(geometry, sort_keys=True)),
            }
            if pixel_weight is not None:
                arrays["pixel_weight"] = np.asarray(pixel_weight, dtype=np.float32)
            if quality_prediction_mask is not None:
                arrays["quality_prediction_mask"] = np.asarray(
                    quality_prediction_mask, dtype=np.uint8
                )
            np.savez_compressed(handle, **arrays)
        temporary.replace(target)

    def metadata(self, version: int | None = None) -> dict:
        path = (
            self.metadata_path
            if version is None
            else self.version_metadata_path(version)
        )
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def write_version_metadata(self, version: int, payload: dict) -> None:
        target = self.version_metadata_path(version)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(target)

    def activate_version(self, version: int) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        version_metadata = self.metadata(version)
        if not version_metadata:
            raise RuntimeError(f"Cannot activate uncommitted cache version {version}")
        payload = {
            "active_version": int(version),
            "cache_version": int(version),
            "teacher_type": version_metadata["teacher_type"],
            "case_ids": version_metadata["case_ids"],
            "num_views": version_metadata["num_views"],
            "prompts": version_metadata["prompts"],
            "standard_prompt": version_metadata["standard_prompt"],
        }
        if "quality_protocol" in version_metadata:
            payload["quality_protocol"] = version_metadata["quality_protocol"]
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
        self.quality_protocol = str(
            dynamic_config.get("quality_protocol", "dynamic_hard_consensus")
        )
        self.amp_enabled = bool(config["model"].get("amp", True)) and (
            runtime.device.type == "cuda"
        )
        self.disable_lora = True
        self.teacher_iteration: int | None = None
        self.case_confidence_rows: dict[str, float] = {}

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

    def _benchmark_gaussian_predictions(
        self,
        volume: torch.Tensor,
        slicers: list[tuple[slice, slice, slice]],
        embedding: torch.Tensor,
        use_uncertainty_views: bool,
    ) -> list[np.ndarray]:
        view_count = (
            int(self.config["method"]["num_views"]) if use_uncertainty_views else 1
        )
        shape = tuple(int(value) for value in volume.shape[-3:])
        accumulation_dtype = np.float16 if self.amp_enabled else np.float32
        torch_dtype = torch.float16 if self.amp_enabled else torch.float32
        accumulations = [np.zeros(shape, dtype=accumulation_dtype) for _ in range(view_count)]
        normalization = np.zeros(shape, dtype=accumulation_dtype)
        gaussian = compute_gaussian(
            self.runtime.patch_size,
            sigma_scale=1 / 8,
            value_scaling_factor=10,
            dtype=torch_dtype,
            device=self.runtime.device,
        ).cpu().numpy()
        with torch.no_grad():
            for slicer in slicers:
                image = self.store.extract_patch(volume, slicer).unsqueeze(0)
                image = image.to(self.runtime.device)
                views = (
                    self.views.make_uncertainty_views(image)
                    if use_uncertainty_views
                    else None
                )
                for view_index in range(view_count):
                    view_image = image if views is None else views[view_index].image
                    with self._autocast():
                        logits = self.runtime.model(view_image, embedding)[0, 0]
                    if views is not None:
                        logits = views[view_index].align(logits.float())
                    patch_logits = logits.to(dtype=torch_dtype).cpu().numpy()
                    accumulations[view_index][slicer] += patch_logits * gaussian
                normalization[slicer] += gaussian
        denominator = np.maximum(normalization, 1e-6)
        return [
            (1.0 / (1.0 + np.exp(-np.clip(accumulation / denominator, -80.0, 80.0))))
            .astype(np.float32)
            for accumulation in accumulations
        ]

    def build_case(
        self,
        case: CaseRecord,
        cache: DynamicHardCache,
        version: int,
        volume: torch.Tensor,
        slicers: list[tuple[slice, slice, slice]],
        case_geometry: dict | None = None,
    ) -> dict:
        image_predictions = []
        was_training = self.runtime.model.training
        self.runtime.model.eval()
        lora_context = (
            lora_disabled(self.runtime.model) if self.disable_lora else nullcontext()
        )
        with torch.no_grad(), lora_context:
            standard_embedding = self.prompt_embeddings[
                str(self.config["dynamic_hard"]["standard_prompt"])
            ]
            if self.quality_protocol == "benchmark_gaussian":
                image_predictions = self._benchmark_gaussian_predictions(
                    volume, slicers, standard_embedding, use_uncertainty_views=True
                )
            else:
                for view_index in range(int(self.config["method"]["num_views"])):
                    patches = []
                    for slicer in slicers:
                        image = self.store.extract_patch(volume, slicer).unsqueeze(0)
                        image = image.to(self.runtime.device)
                        # Match the historical view generator call order exactly.
                        view = self.views.make_uncertainty_views(image)[view_index]
                        with self._autocast():
                            logits = self.runtime.model(view.image, standard_embedding)
                        patches.append(torch.sigmoid(logits.float())[0, 0].cpu().numpy())
                    image_predictions.append(
                        self._stitch(patches, tuple(volume.shape[-3:]), slicers)
                    )

            prompt_predictions = []
            standard_prompt_prediction = None
            for prompt in self.config["dynamic_hard"]["prompts"]:
                if self.quality_protocol == "benchmark_gaussian":
                    stitched = self._benchmark_gaussian_predictions(
                        volume,
                        slicers,
                        self.prompt_embeddings[str(prompt)],
                        use_uncertainty_views=False,
                    )[0]
                else:
                    patches = []
                    for slicer in slicers:
                        image = self.store.extract_patch(volume, slicer).unsqueeze(0)
                        image = image.to(self.runtime.device)
                        with self._autocast():
                            logits = self.runtime.model(
                                image, self.prompt_embeddings[str(prompt)]
                            )
                        patches.append(torch.sigmoid(logits.float())[0, 0].cpu().numpy())
                    stitched = self._stitch(
                        patches, tuple(volume.shape[-3:]), slicers
                    )
                prompt_predictions.append(stitched)
                if str(prompt) == str(self.config["dynamic_hard"]["standard_prompt"]):
                    standard_prompt_prediction = stitched
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
        image_mean = np.mean(np.stack(image_predictions, axis=0), axis=0)
        prompt_mean = np.mean(np.stack(prompt_predictions, axis=0), axis=0)

        def entropy(probability: np.ndarray) -> np.ndarray:
            p = np.clip(probability.astype(np.float32, copy=False), 1e-6, 1.0 - 1e-6)
            return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))

        pixel_weight = np.clip(
            1.0 - 0.5 * (entropy(image_mean) + entropy(prompt_mean)), 0.0, 1.0
        ).astype(np.float32)
        pixel_weight = pixel_weight[None]
        if standard_prompt_prediction is None:
            raise RuntimeError("Dynamic-hard refresh omitted the standard prompt")
        original_prediction_empty = not bool(
            np.any(standard_prompt_prediction >= float(self.config["method"]["probability_threshold"]))
        )
        image_case_confidence = pairwise_hard(
            np.stack(image_predictions, axis=0)
            >= float(self.config["method"]["probability_threshold"])
        )
        prompt_case_confidence = pairwise_hard(
            np.stack(prompt_predictions, axis=0)
            >= float(self.config["method"]["probability_threshold"])
        )
        case_confidence_raw = (
            0.4 * image_case_confidence + 0.6 * prompt_case_confidence
        )
        case_confidence = (
            0.0 if original_prediction_empty else float(case_confidence_raw)
        )
        stat = case.image_path.stat()
        geometry = {
            "case_id": case.name,
            "image_size": int(stat.st_size),
            "image_mtime_ns": int(stat.st_mtime_ns),
            "padded_shape": [int(value) for value in volume.shape[-3:]],
            "roi_shape": [
                int(value)
                for value in (
                    case_geometry.get("roi_shape", volume.shape[-3:])
                    if case_geometry is not None
                    else volume.shape[-3:]
                )
            ],
            "roi_slicer": [
                [int(axis.start or 0), int(axis.stop or 0), int(axis.step or 1)]
                for axis in (
                    case_geometry.get("roi_slicer", tuple(slice(0, int(v)) for v in volume.shape[-3:]))
                    if case_geometry is not None
                    else tuple(slice(0, int(v)) for v in volume.shape[-3:])
                )
            ],
            "cache_version": int(version),
            "teacher_type": (
                "frozen_source" if self.disable_lora else "student_snapshot"
            ),
            "teacher_iteration": self.teacher_iteration,
            "original_prediction_empty": original_prediction_empty,
            "image_case_confidence": float(image_case_confidence),
            "prompt_case_confidence": float(prompt_case_confidence),
            "case_confidence_raw": float(case_confidence_raw),
            "case_confidence": case_confidence,
            "predicted_voxels": int(
                np.count_nonzero(
                    standard_prompt_prediction
                    >= float(self.config["method"]["probability_threshold"])
                )
            ),
        }
        cache.save_case(
            case,
            version,
            joint_probability,
            joint_confidence,
            geometry,
            pixel_weight=pixel_weight,
            quality_prediction_mask=(
                standard_prompt_prediction[None]
                >= float(self.config["method"]["probability_threshold"])
            ),
        )
        self.logger.info(
            "dynamic_hard_cache version=%d case=%s views=%d prompts=%d shape=%s",
            version,
            case.name,
            len(image_predictions),
            len(prompt_predictions),
            tuple(volume.shape[-3:]),
        )
        return geometry

    def build(
        self,
        cases: list[CaseRecord],
        cache: DynamicHardCache,
        version: int,
        teacher_type: str,
        teacher_iteration: int | None = None,
    ) -> None:
        if teacher_type == "frozen_source" and not self.disable_lora:
            raise RuntimeError("Frozen source cache requires LoRA to be disabled")
        if teacher_type == "student_snapshot" and self.disable_lora:
            raise RuntimeError("Student snapshot cache requires LoRA to be enabled")
        self.logger.info(
            "building Dynamic-hard cache version=%d cases=%d teacher=%s",
            version,
            len(cases),
            teacher_type,
        )
        self.teacher_iteration = (
            None if teacher_iteration is None else int(teacher_iteration)
        )
        was_training = self.runtime.model.training
        try:
            for case in cases:
                case_file = cache.case_path(case.name, version)
                if case_file.is_file():
                    try:
                        partial = cache.load_case(case, version)
                        geometry = partial["geometry"]
                        stat = case.image_path.stat()
                        reusable = (
                            geometry.get("case_id") == case.name
                            and geometry.get("cache_version") == int(version)
                            and geometry.get("teacher_type") == teacher_type
                            and geometry.get("teacher_iteration") == self.teacher_iteration
                            and geometry.get("image_size") == int(stat.st_size)
                            and geometry.get("image_mtime_ns") == int(stat.st_mtime_ns)
                        )
                        if reusable:
                            confidence = geometry.get("case_confidence")
                            if confidence is not None:
                                self.case_confidence_rows[case.name] = float(confidence)
                            self.logger.info(
                                "reusing completed Dynamic-hard cache version=%d case=%s",
                                version,
                                case.name,
                            )
                            continue
                    except (OSError, ValueError, KeyError, json.JSONDecodeError):
                        pass
                volume, slicers, case_geometry = self.store.sliding_window_with_geometry(
                    case, self.tile_step_size
                )
                geometry = self.build_case(
                    case, cache, version, volume, slicers, case_geometry
                )
                if "case_confidence" in geometry:
                    self.case_confidence_rows[case.name] = float(
                        geometry["case_confidence"]
                    )
        finally:
            self.runtime.model.train(was_training)
        dynamic = self.config["dynamic_hard"]
        if len(self.case_confidence_rows) == len(cases):
            confidence_path = cache.version_dir(version) / "case_confidences.json"
            temporary = confidence_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(self.case_confidence_rows, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(confidence_path)
        cache.write_version_metadata(
            version,
            {
                "schema_version": 1,
                "cache_version": int(version),
                "teacher_type": teacher_type,
                "num_cases": len(cases),
                "case_ids": sorted(case.name for case in cases),
                "num_views": int(self.config["method"]["num_views"]),
                "prompts": [str(prompt) for prompt in dynamic["prompts"]],
                "standard_prompt": str(dynamic["standard_prompt"]),
                "image_temperature": float(dynamic["image_temperature"]),
                "prompt_temperature": float(dynamic["prompt_temperature"]),
                "refresh_tile_step_size": float(dynamic["refresh_tile_step_size"]),
                "teacher_iteration": self.teacher_iteration,
                "source_checkpoint": source_checkpoint_signature(
                    self.config["model"]
                ),
                **(
                    {"quality_protocol": self.quality_protocol}
                    if "quality_protocol" in dynamic
                    else {}
                ),
            },
        )
        cache.activate_version(version)
