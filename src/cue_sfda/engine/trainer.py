from __future__ import annotations

from contextlib import nullcontext
import json
import random
from pathlib import Path

import numpy as np
import torch

from cue_sfda.data import CaseRecord, TargetVolumeStore
from cue_sfda.engine.checkpoint import load_checkpoint, save_checkpoint
from cue_sfda.evaluation import evaluate
from cue_sfda.method.augmentations import StrongViewGenerator
from cue_sfda.method.dynamic_hard import (
    DynamicHardCache,
    DynamicHardCacheBuilder,
    refresh_ready,
)
from cue_sfda.method.losses import (
    balanced_masked_binary_cross_entropy,
    bernoulli_kl_divergence,
    binary_entropy,
    masked_binary_cross_entropy,
    masked_soft_dice_loss,
)
from cue_sfda.method.refinement import affinity_random_walk
from cue_sfda.method.routing import build_routing_assignments, write_routing_csv
from cue_sfda.method.uncertainty import (
    ChebyshevEstimate,
    ConfidencePartition,
    chebyshev_estimate,
    pseudo_label_quality,
)
from cue_sfda.modeling.voxtell import VoxTellRuntime, load_prompt_embedding
from cue_sfda.modeling.lora import lora_disabled
from cue_sfda.utils import save_json


class CUETrainer:
    def __init__(
        self,
        config: dict,
        runtime: VoxTellRuntime,
        train_cases: list[CaseRecord],
        output_dir: str | Path,
        logger,
        validation_cases: list[CaseRecord] | None = None,
        validation_gt_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.model = runtime.model
        self.embedding = runtime.embedding
        self.device = runtime.device
        self.train_cases = train_cases
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.validation_cases = validation_cases or []
        self.validation_gt_path = validation_gt_path
        data_config = config["data"]
        method_config = config["method"]
        train_config = config["train"]
        self.patch_size = tuple(int(value) for value in data_config["patch_size"])
        self.tile_step_size = float(train_config.get("tile_step_size", 0.5))
        self.store = TargetVolumeStore(self.patch_size)
        seed = int(config["experiment"]["seed"])
        self.generator = torch.Generator().manual_seed(seed)
        self.views = StrongViewGenerator(method_config, seed + 1)
        self.partition = ConfidencePartition(
            quality_threshold=float(method_config["quality_threshold"]),
            momentum=float(method_config["confidence_ema"]),
        )
        self.optimizer = torch.optim.AdamW(
            runtime.trainable_parameters,
            lr=float(train_config["learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
        )
        self.amp_enabled = bool(config["model"].get("amp", True))
        self.amp_enabled = self.amp_enabled and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.start_iteration = 0
        self.best_dice = float("-inf")
        self.best_iteration: int | None = None
        self.best_checkpoint_path: Path | None = None
        self.last_validation_iteration: int | None = None
        self.calibration_foreground_counts: dict[str, int] = {}
        self.skipped_case_names: set[str] = set()
        self.routing_metadata: dict[str, int | float | str] = {}
        self.dynamic_hard_config = config.get("dynamic_hard", {})
        self.dynamic_hard_enabled = (
            str(self.dynamic_hard_config.get("mode", "none")) == "dynamic_hard"
        )
        self.dynamic_hard_cache: DynamicHardCache | None = None
        self.dynamic_prompt_embeddings: dict[str, torch.Tensor] = {}
        self.dynamic_hard_cache_memory: dict[str, np.ndarray | dict] = {}
        self.dynamic_hard_memory_case: str | None = None
        self.refresh_cycle = 0
        self.max_refresh_cycles = int(
            self.dynamic_hard_config.get("max_refresh_cycles", 1)
        )
        self.seen_cases_since_refresh: set[str] = set()
        self.refresh_history: list[dict] = []
        self.patience_counter = 0
        self.stop_iteration: int | None = None
        self.stop_reason: str | None = None
        self.last_training_iteration = 0
        self.branch_sample_counts = {"high": 0, "low": 0}
        self.branch_patch_counts = {"high": 0, "low": 0}
        self.branch_effective_patch_counts = {"high": 0, "low": 0}
        self.branch_skipped_patch_counts = {"high": 0, "low": 0}
        self.dynamic_hard_loss_total = 0.0
        self.dynamic_hard_loss_patches = 0
        if self.dynamic_hard_enabled:
            self.dynamic_hard_cache = DynamicHardCache(
                self.dynamic_hard_config["cache_dir"]
            )
            standard_prompt = str(
                self.dynamic_hard_config["standard_prompt"]
            )
            self.dynamic_prompt_embeddings[standard_prompt] = self.embedding
            for prompt in self.dynamic_hard_config["prompts"]:
                prompt = str(prompt)
                if prompt not in self.dynamic_prompt_embeddings:
                    self.dynamic_prompt_embeddings[prompt] = load_prompt_embedding(
                        config,
                        prompt,
                        self.output_dir / "prompt_cache",
                        self.device,
                    )

    def run(
        self,
        resume: str | Path | None = None,
        confidence_partition: str | Path | None = None,
    ) -> Path:
        if self.dynamic_hard_enabled:
            if confidence_partition is not None:
                raise ValueError(
                    "Dynamic-hard M1 derives its partition from routing; "
                    "--confidence-partition is not supported"
                )
            if resume is not None:
                self._resume(resume)
            self._initialize_routing(
                str(self.config["method"].get("routing_mode", "original"))
            )
            routing_counts = (
                int(self.routing_metadata.get("high", -1)),
                int(self.routing_metadata.get("low", -1)),
                int(self.routing_metadata.get("skip", -1)),
            )
            if len(self.train_cases) != 101 or routing_counts != (59, 42, 0):
                raise RuntimeError(
                    "M1 Dynamic-hard requires 101 routed cases with "
                    f"High=59, Low=42, Skip=0; got {len(self.train_cases)} and "
                    f"{routing_counts}"
                )
            self._ensure_dynamic_hard_cache()
            if resume is None:
                self._save_latest(0)
            elif (
                self.stop_iteration is not None
                and self.start_iteration >= int(self.stop_iteration)
            ):
                latest_path = self.output_dir / "checkpoints" / "latest.pt"
                if not latest_path.is_file():
                    raise RuntimeError(
                        "Dynamic-hard checkpoint is marked early-stopped, but "
                        f"the latest checkpoint is missing: {latest_path}"
                    )
                self.logger.info(
                    "Dynamic-hard run already stopped at iteration %d; "
                    "returning existing latest checkpoint",
                    self.stop_iteration,
                )
                return latest_path
        else:
            if resume is not None:
                self._resume(resume)
            elif confidence_partition is not None:
                self._load_partition(confidence_partition)
        if (
            self.start_iteration == 0
            and bool(self.config["train"].get("validate_at_start", True))
            and self.validation_cases
        ):
            self._validate(0)
        if not self.partition.scores:
            routing_mode = str(self.config["method"].get("routing_mode", "original"))
            if self.config["method"].get("routing_uncertainty_csv"):
                self._initialize_routing(routing_mode)
            else:
                self._calibrate()
            self._save_latest(self.start_iteration)
        method_config = self.config["method"]
        train_config = self.config["train"]
        total_iterations = int(method_config["iterations"])
        save_every = int(train_config["save_every"])
        log_every = int(train_config["log_every"])
        validation_every = int(train_config.get("validation_every", 0))
        high_indices = [
            index
            for index, case in enumerate(self.train_cases)
            if self.partition.is_high_confidence(case.name)
        ]
        low_indices = [
            index
            for index, case in enumerate(self.train_cases)
            if not self.partition.is_high_confidence(case.name)
            and case.name not in self.skipped_case_names
        ]
        if not high_indices and not low_indices:
            raise RuntimeError("Routing excluded every training case")
        self.model.train()
        for iteration in range(self.start_iteration + 1, total_iterations + 1):
            case_index = self._sample_case_index(high_indices, low_indices)
            partner_index = int(
                torch.randint(
                    len(self.train_cases), (1,), generator=self.generator
                ).item()
            )
            case = self.train_cases[case_index]
            partner = self.train_cases[partner_index]
            metrics = self._train_step(case, partner)
            self.last_training_iteration = iteration
            if self.dynamic_hard_enabled:
                branch = str(metrics["branch"])
                self.branch_sample_counts[branch] += 1
                self.branch_patch_counts[branch] += int(metrics["patches"])
                self.branch_effective_patch_counts[branch] += int(
                    metrics["effective_patches"]
                )
                self.branch_skipped_patch_counts[branch] += int(
                    metrics["skipped_patches"]
                )
                self.dynamic_hard_loss_total += float(
                    metrics.get("dynamic_hard_loss", 0.0)
                ) * int(metrics["patches"])
                if branch == "low" and "dynamic_hard_loss" in metrics:
                    self.dynamic_hard_loss_patches += int(metrics["patches"])
                self.seen_cases_since_refresh.add(case.name)
            if iteration == 1 or iteration % log_every == 0:
                if self.dynamic_hard_enabled:
                    self.logger.info(
                        "iteration=%d/%d case=%s patches=%d effective_patches=%d skipped_patches=%d branch=%s loss=%.6f dynamic_hard_loss=%.6f confidence=%.4f raw_foreground=%.6f refined_foreground=%.6f split=%.4f",
                        iteration,
                        total_iterations,
                        case.name,
                        metrics["patches"],
                        metrics["effective_patches"],
                        metrics["skipped_patches"],
                        metrics["branch"],
                        metrics["loss"],
                        metrics.get("dynamic_hard_loss", 0.0),
                        metrics["confidence"],
                        metrics["raw_foreground_fraction"],
                        metrics["refined_foreground_fraction"],
                        self.partition.threshold,
                    )
                else:
                    self.logger.info(
                        "iteration=%d/%d case=%s patches=%d effective_patches=%d skipped_patches=%d branch=%s loss=%.6f confidence=%.4f raw_foreground=%.6f refined_foreground=%.6f split=%.4f",
                        iteration,
                        total_iterations,
                        case.name,
                        metrics["patches"],
                        metrics["effective_patches"],
                        metrics["skipped_patches"],
                        metrics["branch"],
                        metrics["loss"],
                        metrics["confidence"],
                        metrics["raw_foreground_fraction"],
                        metrics["refined_foreground_fraction"],
                        self.partition.threshold,
                    )
            if iteration % save_every == 0 and iteration < total_iterations:
                self._save_latest(iteration)
            should_validate = validation_every > 0
            should_validate = should_validate and iteration % validation_every == 0
            should_validate = should_validate and bool(self.validation_cases)
            if should_validate:
                should_stop = self._validate(iteration)
                if should_stop:
                    self.stop_iteration = iteration
                    self.stop_reason = "early_stopping_patience"
                    self._write_early_stopping()
                    break
            if self.dynamic_hard_enabled and self._dynamic_hard_refresh_ready():
                self._refresh_dynamic_hard_cache(iteration)
        final_iteration = self.stop_iteration or total_iterations
        if self.validation_cases and self.last_validation_iteration != final_iteration:
            self._validate(final_iteration)
        if self.dynamic_hard_enabled:
            if self.stop_iteration is None:
                self.stop_iteration = final_iteration
                self.stop_reason = "max_iterations"
            self._save_latest(final_iteration)
            self._write_early_stopping()
            self._write_refresh_history()
            self._write_training_summary()
            self.logger.info(
                "Saved latest LoRA checkpoint: %s", self.output_dir / "checkpoints/latest.pt"
            )
            return self.output_dir / "checkpoints/latest.pt"
        latest_path = self._save_latest(final_iteration)
        self.logger.info("Saved latest LoRA checkpoint: %s", latest_path)
        return latest_path

    def _load_partition(self, path: str | Path) -> None:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        scores = {str(name): float(value) for name, value in payload["scores"].items()}
        expected = {case.name for case in self.train_cases}
        if set(scores) != expected:
            raise ValueError("Confidence partition does not match training split")
        self.partition.initialize(scores)
        self.calibration_foreground_counts = {
            str(name): int(value)
            for name, value in payload.get("foreground_counts", {}).items()
        }
        self.logger.info("Loaded DWI confidence partition: %s", path)

    def _initialize_routing(self, mode: str) -> None:
        method_config = self.config["method"]
        assignments, scores, metadata = build_routing_assignments(
            [case.name for case in self.train_cases],
            mode=mode,
            uncertainty_csv=method_config["routing_uncertainty_csv"],
            reference_partition=method_config["routing_reference_partition"],
            target_high_count=method_config.get("routing_high_cases"),
        )
        self.partition.initialize(scores)
        self.skipped_case_names = {
            row.case_id for row in assignments if row.assigned_branch == "skip"
        }
        self.routing_metadata = metadata
        write_routing_csv(self.output_dir / "routing.csv", assignments)
        save_json(metadata, self.output_dir / "routing_summary.json")
        self.logger.info(
            "Routing mode=%s score=%s reference_high=%d reference_low=%d "
            "high=%d low=%d skip=%d no_response=%d",
            metadata["mode"], metadata["ranking_score"],
            metadata["reference_high"], metadata["reference_low"],
            metadata["high"], metadata["low"], metadata["skip"],
            metadata["no_response"],
        )

    def _ensure_dynamic_hard_cache(self) -> None:
        if self.dynamic_hard_cache is None:
            raise RuntimeError("Dynamic-hard cache is not configured")
        cases = self.train_cases
        expected_ids = sorted(case.name for case in cases)
        metadata = self.dynamic_hard_cache.metadata()
        expected_prompts = [
            str(prompt) for prompt in self.dynamic_hard_config["prompts"]
        ]
        compatible = (
            set(metadata.get("case_ids", [])) == set(expected_ids)
            and metadata.get("prompts") == expected_prompts
            and int(metadata.get("num_views", -1))
            == int(self.config["method"]["num_views"])
        )
        if not self.dynamic_hard_cache.has_version(0, cases) or not compatible:
            if self.start_iteration > 0 and self.refresh_cycle > 0:
                raise RuntimeError(
                    "Cannot resume Dynamic-hard: the checkpoint's active cache "
                    f"version {self.refresh_cycle} is missing or incompatible"
                )
            builder = DynamicHardCacheBuilder(
                self.config,
                self.runtime,
                self.store,
                StrongViewGenerator(
                    self.config["method"],
                    int(self.config["experiment"]["seed"]) + 10000,
                ),
                self.dynamic_prompt_embeddings,
                self.logger,
            )
            builder.disable_lora = True
            builder.build(cases, self.dynamic_hard_cache, 0, "frozen_source")
            metadata = self.dynamic_hard_cache.metadata()
        if self.start_iteration == 0:
            self.refresh_cycle = 0
            active_version = 0
        else:
            active_version = self.refresh_cycle
            if not self.dynamic_hard_cache.has_version(active_version, cases):
                raise RuntimeError(
                    f"Missing Dynamic-hard cache version {active_version} for resume"
                )
        active_metadata = dict(metadata)
        active_metadata["active_version"] = active_version
        active_metadata["cache_version"] = active_version
        self.dynamic_hard_cache.activate_version(active_metadata)
        save_json(
            {"mode": "dynamic_hard", **active_metadata},
            self.output_dir / "dynamic_hard_cache_metadata.json",
        )
        self._write_refresh_history()
        self.logger.info(
            "Using Dynamic-hard cache=%s version=%d cases=%d",
            self.dynamic_hard_cache.root,
            active_version,
            len(expected_ids),
        )

    def _dynamic_hard_refresh_ready(self) -> bool:
        if not self.dynamic_hard_enabled:
            return False
        expected = {case.name for case in self.train_cases}
        return refresh_ready(
            self.seen_cases_since_refresh,
            expected,
            self.refresh_cycle,
            self.max_refresh_cycles,
        )

    def _refresh_dynamic_hard_cache(self, iteration: int) -> None:
        if self.dynamic_hard_cache is None:
            raise RuntimeError("Dynamic-hard cache is not configured")
        cycle = self.refresh_cycle + 1
        covered = len(self.seen_cases_since_refresh)
        previous_refresh_iteration = (
            int(self.refresh_history[-1]["trigger_iteration"])
            if self.refresh_history
            else 0
        )
        if covered != len(self.train_cases):
            raise RuntimeError(
                "Dynamic-hard refresh was requested before full case coverage"
            )
        builder = DynamicHardCacheBuilder(
            self.config,
            self.runtime,
            self.store,
            StrongViewGenerator(
                self.config["method"],
                int(self.config["experiment"]["seed"]) + 10000 + cycle,
            ),
            self.dynamic_prompt_embeddings,
            self.logger,
        )
        builder.disable_lora = False
        was_training = self.model.training
        try:
            self.model.eval()
            builder.build(
                self.train_cases,
                self.dynamic_hard_cache,
                cycle,
                "student_snapshot",
            )
        finally:
            self.model.train(was_training)
        self.refresh_history.append(
            {
                "refresh_cycle": cycle,
                "trigger_iteration": int(iteration),
                "covered_cases": covered,
                "iterations_since_previous_refresh": int(
                    iteration - previous_refresh_iteration
                ),
                "teacher_type": "student_snapshot",
            }
        )
        self.refresh_cycle = cycle
        self.seen_cases_since_refresh.clear()
        self.dynamic_hard_cache_memory.clear()
        self.dynamic_hard_memory_case = None
        metadata = self.dynamic_hard_cache.metadata()
        save_json(
            {"mode": "dynamic_hard", **metadata},
            self.output_dir / "dynamic_hard_cache_metadata.json",
        )
        self._write_refresh_history()
        self.logger.info(
            "Dynamic-hard cache refresh complete cycle=%d iteration=%d",
            cycle,
            iteration,
        )

    def _write_refresh_history(self) -> None:
        if not self.dynamic_hard_enabled:
            return
        save_json(
            {
                "cache_version": int(self.refresh_cycle),
                "refresh_history": self.refresh_history,
                "seen_cases_since_refresh": sorted(self.seen_cases_since_refresh),
            },
            self.output_dir / "refresh_history.json",
        )

    def _write_early_stopping(self) -> None:
        if not self.dynamic_hard_enabled:
            return
        save_json(
            {
                "best_dice": (
                    None if self.best_dice == float("-inf") else self.best_dice
                ),
                "best_iteration": self.best_iteration,
                "best_checkpoint": (
                    str(self.best_checkpoint_path)
                    if self.best_checkpoint_path is not None
                    else None
                ),
                "patience_counter": self.patience_counter,
                "early_stopping_patience": int(
                    self.config["train"].get("early_stopping_patience", 0)
                ),
                "stop_iteration": self.stop_iteration,
                "stop_reason": self.stop_reason,
            },
            self.output_dir / "early_stopping.json",
        )

    def _write_training_summary(self) -> None:
        if not self.dynamic_hard_enabled:
            return
        save_json(
            {
                "iterations_completed": int(self.last_training_iteration),
                "branch_sample_counts": self.branch_sample_counts,
                "branch_patch_counts": self.branch_patch_counts,
                "branch_effective_supervision_patches": (
                    self.branch_effective_patch_counts
                ),
                "branch_ineffective_patches": self.branch_skipped_patch_counts,
                "dynamic_hard_loss_mean": (
                    self.dynamic_hard_loss_total
                    / max(self.dynamic_hard_loss_patches, 1)
                ),
                "dynamic_hard_loss_patches": self.dynamic_hard_loss_patches,
            },
            self.output_dir / "training_summary.json",
        )

    def _trainer_checkpoint_state(self) -> dict:
        return {
            "dynamic_hard_enabled": self.dynamic_hard_enabled,
            "refresh_cycle": self.refresh_cycle,
            "seen_cases_since_refresh": sorted(self.seen_cases_since_refresh),
            "refresh_history": self.refresh_history,
            "patience_counter": self.patience_counter,
            "stop_iteration": self.stop_iteration,
            "stop_reason": self.stop_reason,
            "last_training_iteration": self.last_training_iteration,
            "branch_sample_counts": self.branch_sample_counts,
            "branch_patch_counts": self.branch_patch_counts,
            "branch_effective_patch_counts": self.branch_effective_patch_counts,
            "branch_skipped_patch_counts": self.branch_skipped_patch_counts,
            "dynamic_hard_loss_total": self.dynamic_hard_loss_total,
            "dynamic_hard_loss_patches": self.dynamic_hard_loss_patches,
        }

    def _sample_case_index(
        self,
        high_indices: list[int],
        low_indices: list[int],
    ) -> int:
        probability = float(
            self.config["method"].get("high_quality_sampling_probability", 0.5)
        )
        use_high = bool(torch.rand((), generator=self.generator) < probability)
        candidates = high_indices if use_high else low_indices
        if not candidates:
            candidates = low_indices if use_high else high_indices
        selected = int(
            torch.randint(len(candidates), (1,), generator=self.generator).item()
        )
        return candidates[selected]

    def _calibrate(self) -> None:
        requested = int(self.config["method"].get("calibration_cases", 0))
        count = (
            len(self.train_cases)
            if requested <= 0
            else min(requested, len(self.train_cases))
        )
        scores = {}
        foreground_counts = {}
        minimum_support = int(
            self.config["method"].get("min_case_pseudo_foreground_voxels", 1)
        )
        self.logger.info("Calibrating Chebyshev confidence on %d target cases", count)
        self.model.eval()
        for index, case in enumerate(self.train_cases[:count]):
            partner = self.train_cases[(index + 1) % len(self.train_cases)]
            case_volume, case_slicers = self.store.sliding_window(
                case,
                self.tile_step_size,
            )
            partner_volume, partner_slicers = self.store.sliding_window(
                partner,
                self.tile_step_size,
            )
            raw_count = 0
            filtered_count = 0
            for patch_index, case_slicer in enumerate(case_slicers):
                partner_slicer = partner_slicers[patch_index % len(partner_slicers)]
                image = self._patch(case_volume, case_slicer)
                partner_image = self._patch(partner_volume, partner_slicer)
                estimate = self._estimate(image, partner_image)
                raw = estimate.raw_probability > float(
                    self.config["method"]["probability_threshold"]
                )
                raw_count += int(raw.sum().item())
                filtered_count += int((raw & estimate.confidence_mask).sum().item())
            foreground_counts[case.name] = raw_count
            scores[case.name] = pseudo_label_quality(
                raw_count,
                filtered_count,
                minimum_foreground_voxels=minimum_support,
            )
        if count < len(self.train_cases):
            default_score = float(torch.tensor(list(scores.values())).median().item())
            for case in self.train_cases[count:]:
                scores[case.name] = default_score
                foreground_counts[case.name] = minimum_support
        self.partition.initialize(scores)
        self.calibration_foreground_counts = foreground_counts
        high_count = sum(
            self.partition.is_high_confidence(case.name) for case in self.train_cases
        )
        self.logger.info(
            "Confidence split threshold=%.6f minimum_foreground=%d high=%d low=%d",
            self.partition.threshold,
            minimum_support,
            high_count,
            len(self.train_cases) - high_count,
        )
        save_json(
            {
                "threshold": self.partition.threshold,
                "scores": self.partition.scores,
                "minimum_foreground_voxels": minimum_support,
                "foreground_counts": foreground_counts,
            },
            self.output_dir / "confidence_partition.json",
        )

    def _train_step(
        self, case: CaseRecord, partner: CaseRecord
    ) -> dict[str, float | str | int]:
        high_confidence = self.partition.is_high_confidence(case.name)
        case_volume, case_slicers = self.store.sliding_window(
            case,
            self.tile_step_size,
        )
        partner_volume, partner_slicers = self.store.sliding_window(
            partner,
            self.tile_step_size,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        confidence_sum = 0.0
        raw_foreground_sum = 0
        refined_foreground_sum = 0
        dynamic_hard_loss_sum = 0.0
        voxel_sum = 0
        effective_patch_count = 0
        skipped_patch_count = 0
        patch_count = len(case_slicers)
        for patch_index, case_slicer in enumerate(case_slicers):
            partner_slicer = partner_slicers[patch_index % len(partner_slicers)]
            image = self._patch(case_volume, case_slicer)
            partner_image = self._patch(partner_volume, partner_slicer)
            if high_confidence:
                loss, patch_metrics = self._high_confidence_loss(image)
                branch = "high"
            elif self.dynamic_hard_enabled:
                loss, patch_metrics = self._low_confidence_loss(
                    image, partner_image, case, case_slicer
                )
                branch = "low"
            else:
                loss, patch_metrics = self._low_confidence_loss(image, partner_image)
                branch = "low"
            if not torch.isfinite(loss):
                message = f"Non-finite loss for {case.name} patch {patch_index}"
                raise FloatingPointError(message)
            self.scaler.scale(loss).backward()
            loss_sum += float(loss.detach().item())
            confidence_sum += self.partition.scores[case.name]
            raw_foreground_sum += int(patch_metrics["raw_foreground_voxels"])
            refined_foreground_sum += int(
                patch_metrics["refined_foreground_voxels"]
            )
            dynamic_hard_loss_sum += float(
                patch_metrics.get("dynamic_hard_loss", 0.0)
            )
            voxel_sum += int(patch_metrics["voxels"])
            if bool(patch_metrics["effective"]):
                effective_patch_count += 1
            else:
                skipped_patch_count += 1
        self.scaler.unscale_(self.optimizer)
        gradient_denominator = max(effective_patch_count, 1)
        for parameter in self.runtime.trainable_parameters:
            if parameter.grad is not None:
                parameter.grad.div_(gradient_denominator)
        max_norm = float(self.config["train"]["max_grad_norm"])
        torch.nn.utils.clip_grad_norm_(self.runtime.trainable_parameters, max_norm)
        if effective_patch_count > 0:
            self.scaler.step(self.optimizer)
        self.scaler.update()
        case_confidence = confidence_sum / patch_count
        if bool(self.config["method"].get("update_confidence_partition", False)):
            self.partition.update(case.name, case_confidence)
        metrics = {
            "branch": branch,
            "patches": patch_count,
            "effective_patches": effective_patch_count,
            "skipped_patches": skipped_patch_count,
            "loss": loss_sum / gradient_denominator,
            "confidence": case_confidence,
            "raw_foreground_fraction": raw_foreground_sum / max(voxel_sum, 1),
            "refined_foreground_fraction": (
                refined_foreground_sum / max(voxel_sum, 1)
            ),
        }
        if self.dynamic_hard_enabled:
            metrics["dynamic_hard_loss"] = dynamic_hard_loss_sum / max(
                patch_count, 1
            )
        return metrics

    def _estimate(
        self, image: torch.Tensor, partner: torch.Tensor
    ) -> ChebyshevEstimate:
        aligned = []
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            source_only = bool(
                self.config["method"].get("source_only_confidence", True)
            )
            context = lora_disabled(self.model) if source_only else nullcontext()
            with context:
                with self._autocast():
                    raw_logits = self.model(image, self.embedding)
                raw_probability = torch.sigmoid(raw_logits.float())
                probability_threshold = float(
                    self.config["method"]["probability_threshold"]
                )
                if not bool((raw_probability > probability_threshold).any()):
                    ones = torch.ones_like(raw_probability)
                    return ChebyshevEstimate(
                        raw_probability=raw_probability,
                        mean_probability=raw_probability,
                        standard_deviation=torch.zeros_like(raw_probability),
                        error_bound=ones,
                        reliable_mask=torch.zeros_like(raw_probability, dtype=torch.bool),
                        confidence_mask=torch.zeros_like(raw_probability, dtype=torch.bool),
                        case_confidence=0.0,
                    )
                generated_views = self.views.make_uncertainty_views(image)
                for view in generated_views:
                    with self._autocast():
                        logits = self.model(view.image, self.embedding)
                    probability = torch.sigmoid(logits.float())
                    aligned.append(view.align(probability))
        self.model.train(was_training)
        probabilities = torch.stack(aligned, dim=0)
        method_config = self.config["method"]
        return chebyshev_estimate(
            probabilities,
            raw_probability=raw_probability,
            probability_threshold=float(method_config["probability_threshold"]),
            bound_threshold=float(method_config["chebyshev_threshold"]),
        )

    def _high_confidence_loss(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, int | bool]]:
        method_config = self.config["method"]
        teacher_image = image
        self.model.eval()
        with torch.no_grad():
            with lora_disabled(self.model):
                with self._autocast():
                    teacher_logits, features = self.model(
                        teacher_image,
                        self.embedding,
                        return_features=True,
                    )
            teacher_probability = torch.sigmoid(teacher_logits.float())
            threshold = float(method_config["probability_threshold"])
            raw_voxels = int((teacher_probability >= threshold).sum().item())
            propagated = affinity_random_walk(
                probability=teacher_probability,
                features=features,
                image=teacher_image,
                output_size=tuple(teacher_probability.shape[-3:]),
                graph_size=tuple(
                    int(value) for value in method_config["random_walk_size"]
                ),
                steps=int(method_config["random_walk_steps"]),
                alpha=float(method_config["random_walk_alpha"]),
                temperature=float(method_config["affinity_temperature"]),
                score_pooling=str(
                    method_config.get("random_walk_score_pooling", "average")
                ),
            )
            refined = propagated.clamp(0.0, 1.0)
            if bool(method_config.get("preserve_raw_foreground", False)):
                # The paper's 2-D full-resolution graph has no pooling stage.
                # Preserve source seeds in the memory-bounded 3-D extension so
                # adaptive pooling cannot erase small tumor pseudo-labels.
                refined = torch.maximum(refined, teacher_probability)
            del features
            del teacher_logits
            del teacher_probability
            del propagated
        self.model.train()
        with self._autocast():
            student_logits = self.model(teacher_image, self.embedding)
        student_probability = torch.sigmoid(student_logits.float())
        # Once a case passes Eq. (11), the paper supervises the complete
        # affinity-refined pseudo-label rather than masking it a second time.
        weight = torch.ones_like(refined)
        pseudo_voxels = int((refined >= threshold).sum().item())
        minimum = int(method_config.get("min_pseudo_foreground_voxels", 8))
        patch_metrics = {
            "raw_foreground_voxels": raw_voxels,
            "refined_foreground_voxels": pseudo_voxels,
            "voxels": refined.numel(),
            "effective": pseudo_voxels >= minimum,
        }
        if pseudo_voxels < minimum:
            return student_logits.float().sum() * 0.0, patch_metrics
        if bool(self.config["train"].get("balance_high_bce", True)):
            bce = balanced_masked_binary_cross_entropy(
                student_probability, refined, weight
            )
        else:
            foreground_weight = float(
                self.config["train"].get("high_foreground_weight", 5.0)
            )
            weight = weight * (1.0 + (foreground_weight - 1.0) * refined.detach())
            bce = masked_binary_cross_entropy(student_probability, refined, weight)
        dice = masked_soft_dice_loss(student_probability, refined, weight)
        train_config = self.config["train"]
        loss = (
            float(train_config["high_bce_weight"]) * bce
            + float(train_config["high_dice_weight"]) * dice
        )
        return loss, patch_metrics

    def _low_confidence_loss(
        self,
        image: torch.Tensor,
        partner: torch.Tensor,
        case: CaseRecord | None = None,
        case_slicer: tuple[slice, slice, slice] | None = None,
    ) -> tuple[torch.Tensor, dict[str, int | bool]]:
        view_count = int(self.config["method"].get("low_quality_views", 3))
        strong_views = self.views.make_views(image, partner, count=view_count + 2)[1:]
        prototype_views = strong_views[:view_count]
        student_view = strong_views[-1]
        self.model.eval()
        aligned_probabilities = []
        with torch.no_grad():
            for view in prototype_views:
                with self._autocast():
                    teacher_logits = self.model(view.image, self.embedding)
                probability = torch.sigmoid(teacher_logits.float())
                aligned_probabilities.append(view.align(probability))
        prototype = torch.stack(aligned_probabilities).mean(dim=0)
        self.model.train()
        with self._autocast():
            student_logits = self.model(student_view.image, self.embedding)
        student_probability = torch.sigmoid(student_logits.float())
        student_probability = student_view.align(student_probability)
        train_config = self.config["train"]
        consistency = bernoulli_kl_divergence(
            prototype,
            student_probability,
            balanced=bool(train_config.get("balance_low_consistency", True)),
        )
        entropy = binary_entropy(student_probability)
        loss = (
            float(train_config["low_consistency_weight"]) * consistency
            + float(train_config["low_entropy_weight"]) * entropy
        )
        dynamic_hard_loss = 0.0
        if self.dynamic_hard_enabled:
            if case is None or case_slicer is None:
                raise ValueError(
                    "Dynamic-hard supervision requires case and patch geometry"
                )
            target, confidence = self._dynamic_hard_patch(case, case_slicer)
            confidence_threshold = float(
                self.dynamic_hard_config.get("confidence_threshold", 0.0)
            )
            if confidence_threshold > 0.0:
                confidence = confidence * (confidence >= confidence_threshold).float()
            dynamic_hard_loss_tensor = masked_binary_cross_entropy(
                student_probability,
                target.detach(),
                confidence.detach().clamp_min(1e-3),
            )
            dynamic_hard_loss = float(dynamic_hard_loss_tensor.detach().item())
            loss = loss + float(
                self.dynamic_hard_config.get("lambda_consensus", 0.05)
            ) * dynamic_hard_loss_tensor
        threshold = float(self.config["method"]["probability_threshold"])
        prototype_voxels = int((prototype >= threshold).sum().item())
        patch_metrics = {
            "raw_foreground_voxels": prototype_voxels,
            "refined_foreground_voxels": prototype_voxels,
            "voxels": prototype.numel(),
            "effective": True,
        }
        if self.dynamic_hard_enabled:
            patch_metrics["dynamic_hard_loss"] = dynamic_hard_loss
        return loss, patch_metrics

    def _dynamic_hard_patch(
        self,
        case: CaseRecord,
        slicer: tuple[slice, slice, slice],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dynamic_hard_cache is None:
            raise RuntimeError("Dynamic-hard cache is not configured")
        if self.dynamic_hard_memory_case != case.name:
            self.dynamic_hard_cache_memory = self.dynamic_hard_cache.load_case(
                case, self.refresh_cycle
            )
            self.dynamic_hard_memory_case = case.name
        probability = self.dynamic_hard_cache_memory["joint_probability"]
        confidence = self.dynamic_hard_cache_memory["joint_voxel_confidence"]
        geometry = self.dynamic_hard_cache_memory["geometry"]
        expected_shape = tuple(int(value) for value in geometry["padded_shape"])
        if tuple(probability.shape[-3:]) != expected_shape:
            raise ValueError(f"Dynamic-hard cache geometry mismatch: {case.name}")
        target_patch = np.ascontiguousarray(
            probability[(slice(None), *slicer)]
        )
        confidence_patch = np.ascontiguousarray(
            confidence[(slice(None), *slicer)]
        )
        target = torch.from_numpy(target_patch).unsqueeze(0).to(self.device)
        confidence_tensor = torch.from_numpy(confidence_patch).unsqueeze(0).to(
            self.device
        )
        if tuple(target.shape) != tuple(confidence_tensor.shape):
            raise ValueError(f"Dynamic-hard target/confidence mismatch: {case.name}")
        if target.shape[-3:] != self.patch_size:
            raise ValueError(
                f"Dynamic-hard patch has unexpected shape for {case.name}: "
                f"{tuple(target.shape)}"
            )
        return target, confidence_tensor

    def _patch(
        self,
        volume: torch.Tensor,
        slicer: tuple[slice, slice, slice],
    ) -> torch.Tensor:
        patch = self.store.extract_patch(volume, slicer)
        return patch.unsqueeze(0).to(self.device, non_blocking=True)

    def _validate(self, iteration: int) -> bool:
        if self.validation_gt_path is None:
            raise ValueError("Periodic validation requires --gt-path")
        validation_dir = self.output_dir / "validation" / f"iteration_{iteration:06d}"
        self.logger.info("Running validation at iteration %d", iteration)
        was_training = self.model.training
        summary = evaluate(
            self.config,
            self.runtime,
            self.validation_cases,
            self.validation_gt_path,
            validation_dir,
            self.logger,
            save_predictions=bool(
                self.config["train"].get("save_validation_predictions", True)
            ),
        )
        self.model.train(was_training)
        self.last_validation_iteration = iteration
        mean_dice = float(summary["mean_dice"])
        if mean_dice > self.best_dice:
            self.patience_counter = 0
            self._save_best(iteration, mean_dice)
        elif int(self.config["train"].get("early_stopping_patience", 0)) > 0:
            self.patience_counter += 1
        patience = int(self.config["train"].get("early_stopping_patience", 0))
        should_stop = patience > 0 and self.patience_counter >= patience
        if should_stop:
            self.stop_iteration = iteration
            self.stop_reason = "early_stopping_patience"
        self._save_latest(iteration)
        self.logger.info(
            "validation_iteration=%d mean_dice=%.6f mean_iou=%.6f",
            iteration,
            summary["mean_dice"],
            summary["mean_iou"],
        )
        self._write_early_stopping()
        return should_stop

    def _autocast(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _save(self, path: Path, iteration: int) -> Path:
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            iteration=iteration,
            config=self.config,
            confidence_scores=self.partition.scores,
            confidence_threshold=self.partition.threshold,
            best_dice=(None if self.best_dice == float("-inf") else self.best_dice),
            best_iteration=self.best_iteration,
            trainer_generator_state=self.generator.get_state(),
            view_generator_state=self.views.generator.get_state(),
            view_python_generator_state=self.views.python_generator.getstate(),
            scaler=self.scaler,
            trainer_state=(
                self._trainer_checkpoint_state()
                if self.dynamic_hard_enabled
                else None
            ),
        )
        self._write_refresh_history()
        self._write_early_stopping()
        self._write_training_summary()
        return path

    def _save_latest(self, iteration: int) -> Path:
        path = self.output_dir / "checkpoints" / "latest.pt"
        return self._save(path, iteration)

    def _save_best(self, iteration: int, mean_dice: float) -> Path:
        checkpoint_dir = self.output_dir / "checkpoints"
        path = checkpoint_dir / (
            f"best_iteration_{iteration:06d}_dice_{mean_dice:.6f}.pt"
        )
        previous_paths = list(checkpoint_dir.glob("best_iteration_*_dice_*.pt"))
        self.best_dice = float(mean_dice)
        self.best_iteration = int(iteration)
        self.best_checkpoint_path = path
        self._save(path, iteration)
        for previous_path in previous_paths:
            if previous_path != path and previous_path.is_file():
                previous_path.unlink()
        self.logger.info(
            "Saved best LoRA checkpoint: %s (iteration=%d mean_dice=%.6f)",
            path,
            iteration,
            mean_dice,
        )
        return path

    def _resume(self, path: str | Path) -> None:
        checkpoint = load_checkpoint(path, self.model, self.optimizer, self.scaler)
        configured_lr = float(self.config["train"]["learning_rate"])
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = configured_lr
        self.start_iteration = int(checkpoint.get("iteration", 0))
        training_state = checkpoint.get("trainer_state", {})
        self.refresh_cycle = int(training_state.get("refresh_cycle", 0))
        self.seen_cases_since_refresh = set(
            str(name) for name in training_state.get("seen_cases_since_refresh", [])
        )
        self.refresh_history = list(training_state.get("refresh_history", []))
        self.patience_counter = int(training_state.get("patience_counter", 0))
        self.stop_iteration = training_state.get("stop_iteration")
        self.stop_reason = training_state.get("stop_reason")
        self.last_training_iteration = int(
            training_state.get("last_training_iteration", self.start_iteration)
        )
        for field, target in (
            ("branch_sample_counts", self.branch_sample_counts),
            ("branch_patch_counts", self.branch_patch_counts),
            ("branch_effective_patch_counts", self.branch_effective_patch_counts),
            ("branch_skipped_patch_counts", self.branch_skipped_patch_counts),
        ):
            for branch, value in training_state.get(field, {}).items():
                if branch in target:
                    target[branch] = int(value)
        self.dynamic_hard_loss_total = float(
            training_state.get("dynamic_hard_loss_total", 0.0)
        )
        self.dynamic_hard_loss_patches = int(
            training_state.get("dynamic_hard_loss_patches", 0)
        )
        best_dice = checkpoint.get("best_dice")
        best_iteration = checkpoint.get("best_iteration")
        if best_dice is not None:
            self.best_dice = float(best_dice)
        if best_iteration is not None:
            self.best_iteration = int(best_iteration)
            matches = list(
                (self.output_dir / "checkpoints").glob(
                    f"best_iteration_{self.best_iteration:06d}_dice_*.pt"
                )
            )
            if matches:
                self.best_checkpoint_path = matches[0]
        scores = checkpoint.get("confidence_scores", {})
        if scores:
            self.partition.initialize(scores)
        rng_state = checkpoint.get("rng_state")
        if rng_state:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch_cpu"])
            cuda_states = rng_state.get("torch_cuda", [])
            if torch.cuda.is_available() and cuda_states:
                torch.cuda.set_rng_state_all(cuda_states)
            generator_state = rng_state.get("trainer_generator")
            if generator_state is not None:
                self.generator.set_state(generator_state)
            view_state = rng_state.get("view_generator")
            if view_state is not None:
                self.views.generator.set_state(view_state)
            view_python_state = rng_state.get("view_python_generator")
            if view_python_state is not None:
                self.views.python_generator.setstate(view_python_state)
        else:
            self.logger.warning(
                "Checkpoint has no RNG state; stochastic continuation is not exact"
            )
        self.logger.info("Resumed from %s at iteration %d", path, self.start_iteration)
