from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from nnunetv2.inference.sliding_window_prediction import compute_gaussian

from cue_sfda.data import CaseRecord, TargetVolumeStore
from cue_sfda.engine.checkpoint import load_checkpoint, save_checkpoint
from cue_sfda.evaluation import evaluate
from cue_sfda.method.augmentations import StrongViewGenerator
from cue_sfda.method.dynamic_hard import (
    DynamicHardCache,
    DynamicHardCacheBuilder,
    dynamic_hard_consensus_loss,
    refresh_ready,
    sample_coverage_index,
    source_checkpoint_signature,
)
from cue_sfda.high_selection import (
    MODES as HIGH_SELECTION_MODES,
    build_semantic_cache,
    flatten_gradients,
    relative_subspace_error,
)
from cue_sfda.high_selection.base import should_select_high
from cue_sfda.high_selection.amp import (
    conditional_scaler_step,
    gradients_are_finite,
    scaler_will_skip_step,
)
from cue_sfda.high_selection import cross_case_validation, gradient_agreement, semantic_update_validation
from cue_sfda.high_selection.feature_extractor import foreground_d5_tokens
from cue_sfda.high_selection.reference_bank import ReferenceBank
from cue_sfda.method.losses import (
    balanced_masked_binary_cross_entropy,
    bernoulli_kl_divergence,
    binary_entropy,
    masked_binary_cross_entropy,
    masked_soft_dice_loss,
)
from cue_sfda.method.liver_adaptation import (
    build_liver_region_masks,
    fused_volume_constraint_gradient,
    patch_vjp_coefficient,
    partitioned_teacher_loss,
    remote_background_loss,
    teacher_anchor_loss,
)
from cue_sfda.method.refinement import affinity_random_walk
from cue_sfda.method.source_volume import SourceVolumeCache, SourceVolumePrediction
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        self.liver_config = dict(config.get("liver_adaptation", {}))
        self.liver_experiment = str(self.liver_config.get("experiment", "baseline"))
        self.prompt_cache_dir = Path(
            self.liver_config.get("prompt_cache_dir", self.output_dir / "prompt_cache")
        )
        self.source_volume_cache = (
            SourceVolumeCache(
                self.liver_config["teacher_cache_dir"],
                str(config["model"]["prompt"]),
                config["model"],
            )
            if self.liver_config
            else None
        )
        self.liver_loss_totals: dict[str, float] = {}
        self.liver_loss_counts: dict[str, int] = {}
        self.liver_context_case_name: str | None = None
        self.liver_context_memory: dict | None = None
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
        self.high_selection_config = dict(config.get("high_selection", {}))
        self.high_selection_mode = str(self.high_selection_config.get("mode", "none"))
        if self.high_selection_mode not in HIGH_SELECTION_MODES:
            raise ValueError(f"invalid high_selection.mode: {self.high_selection_mode}")
        self.high_reference_bank = ReferenceBank()
        self.high_selection_refs = self.high_reference_bank.gradients
        self.high_selection_features = self.high_reference_bank.features
        self.high_selection_cache_iteration: int | None = None
        self.high_selection_attempts = self.high_selection_accepts = self.high_selection_rejects = 0
        self.high_selection_pending_row: dict | None = None
        self.high_selection_rng = random.Random(int(config["experiment"]["seed"]) + 7319)
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
        self.coverage_mode = False
        self.refresh_history: list[dict] = []
        self.patience_counter = 0
        self.stop_iteration: int | None = None
        self.stop_reason: str | None = None
        self.last_training_iteration = 0
        self.branch_sample_counts = {"high": 0, "low": 0}
        self.branch_patch_counts = {"high": 0, "low": 0}
        self.branch_effective_patch_counts = {"high": 0, "low": 0}
        self.branch_skipped_patch_counts = {"high": 0, "low": 0}
        self.optimizer_step_count = 0
        self.dynamic_hard_loss_total = 0.0
        self.dynamic_hard_loss_patches = 0
        if self.dynamic_hard_enabled:
            self.dynamic_hard_cache = DynamicHardCache(
                self.dynamic_hard_config["cache_dir"]
            )
            for prompt in self.dynamic_hard_config["prompts"]:
                prompt = str(prompt)
                if prompt not in self.dynamic_prompt_embeddings:
                    self.dynamic_prompt_embeddings[prompt] = load_prompt_embedding(
                        config,
                        prompt,
                        self.prompt_cache_dir,
                        self.device,
                    )

    def run(
        self,
        resume: str | Path | None = None,
        confidence_partition: str | Path | None = None,
        coverage_completion: bool = False,
    ) -> Path:
        if coverage_completion and not self.dynamic_hard_enabled:
            raise ValueError("Coverage completion is only valid for Dynamic-hard")
        if self.dynamic_hard_enabled:
            if coverage_completion and resume is None:
                raise ValueError("Coverage completion requires a resumed run")
            if confidence_partition is not None:
                raise ValueError(
                    "Dynamic-hard M1 derives its partition from routing; "
                    "--confidence-partition is not supported"
                )
            if resume is not None:
                self._resume(resume)
            if coverage_completion:
                if self.refresh_cycle >= self.max_refresh_cycles:
                    raise ValueError(
                        "Coverage completion can only be enabled before the "
                        "configured Dynamic-hard refresh"
                    )
                self.coverage_mode = True
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
            if resume is not None and (
                self.seen_cases_since_refresh
                == {case.name for case in self.train_cases}
            ):
                if self._dynamic_hard_refresh_ready():
                    self._refresh_dynamic_hard_cache(self.start_iteration)
                    self._save_latest(self.start_iteration)
                elif self.coverage_mode:
                    self.coverage_mode = False
                    self._save_latest(self.start_iteration)
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
                newly_covered = case.name not in self.seen_cases_since_refresh
                self.seen_cases_since_refresh.add(case.name)
                if (
                    newly_covered
                    and len(self.seen_cases_since_refresh) == len(self.train_cases)
                ):
                    self.logger.info(
                        "Dynamic-hard coverage complete pass=%d iteration=%d "
                        "cases=%d refresh_available=%s",
                        self.refresh_cycle + 1,
                        iteration,
                        len(self.seen_cases_since_refresh),
                        self.refresh_cycle < self.max_refresh_cycles,
                    )
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
            if self.liver_config and iteration % 50 == 0:
                self.logger.info(
                    "liver_loss_components iteration=%d experiment=%s high_bce=%.6f "
                    "high_dice=%.6f volume=%.6f volume_ratio=%.5f remote_background=%.6f "
                    "regions=%.6f anchor=%.6f",
                    iteration,
                    self.liver_experiment,
                    metrics.get("high_bce_loss", 0.0),
                    metrics.get("high_dice_loss", 0.0),
                    metrics.get("volume_constraint_loss", 0.0),
                    metrics.get("student_teacher_volume_ratio", float("nan")),
                    metrics.get("remote_background_loss", 0.0),
                    metrics.get("region_loss", 0.0),
                    metrics.get("anchor_loss", 0.0),
                )
            if (
                not self.dynamic_hard_enabled
                and iteration % save_every == 0
                and iteration < total_iterations
            ):
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
            refreshed = False
            coverage_complete = self.dynamic_hard_enabled and (
                self.seen_cases_since_refresh
                == {case.name for case in self.train_cases}
            )
            if (
                coverage_complete
                and self.coverage_mode
                and not self._dynamic_hard_refresh_ready()
            ):
                self.coverage_mode = False
                self._save_latest(iteration)
                refreshed = True
                self.logger.info(
                    "Dynamic-hard resumed coverage complete pass=%d iteration=%d; "
                    "ordinary sampler restored",
                    self.refresh_cycle + 1,
                    iteration,
                )
            elif self.dynamic_hard_enabled and self._dynamic_hard_refresh_ready():
                self._refresh_dynamic_hard_cache(iteration)
                # Persist the new active cache version and the cleared coverage
                # set together, so a resume cannot roll back across refresh.
                self._save_latest(iteration)
                refreshed = True
            if (
                iteration % save_every == 0
                and iteration < total_iterations
                and not refreshed
                and self.dynamic_hard_enabled
            ):
                self._save_latest(iteration)
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
        expected_coverage = {case.name for case in cases}
        if not self.seen_cases_since_refresh.issubset(expected_coverage):
            raise RuntimeError("Checkpoint coverage contains cases outside the training split")
        if self.refresh_cycle != len(self.refresh_history):
            raise RuntimeError(
                "Dynamic-hard checkpoint refresh count does not match its history"
            )
        if self.refresh_cycle > self.max_refresh_cycles:
            raise RuntimeError("Dynamic-hard checkpoint exceeds configured refresh count")

        active_version = self.refresh_cycle if self.start_iteration > 0 else 0
        expected_teacher = "frozen_source" if active_version == 0 else "student_snapshot"
        expected_teacher_iteration = (
            None
            if active_version == 0
            else int(self.refresh_history[-1]["trigger_iteration"])
        )
        expected_metadata = {
            "schema_version": 1,
            "cache_version": active_version,
            "teacher_type": expected_teacher,
            "case_ids": expected_ids,
            "num_cases": len(expected_ids),
            "num_views": int(self.config["method"]["num_views"]),
            "prompts": [str(prompt) for prompt in self.dynamic_hard_config["prompts"]],
            "standard_prompt": str(self.dynamic_hard_config["standard_prompt"]),
            "image_temperature": float(self.dynamic_hard_config["image_temperature"]),
            "prompt_temperature": float(self.dynamic_hard_config["prompt_temperature"]),
            "refresh_tile_step_size": float(
                self.dynamic_hard_config["refresh_tile_step_size"]
            ),
            "teacher_iteration": expected_teacher_iteration,
            "source_checkpoint": source_checkpoint_signature(
                self.config["model"]
            ),
        }
        try:
            version_metadata = self.dynamic_hard_cache.metadata(active_version)
        except (OSError, json.JSONDecodeError):
            version_metadata = {}
        compatible = all(
            version_metadata.get(key) == value
            for key, value in expected_metadata.items()
        )
        version_complete = self.dynamic_hard_cache.has_version(active_version, cases)
        if compatible and version_complete:
            try:
                for case in cases:
                    self.dynamic_hard_cache.load_case(case, active_version)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                compatible = False
        if not compatible or not version_complete:
            if self.start_iteration > 0:
                raise RuntimeError(
                    "Cannot resume Dynamic-hard: the checkpoint's active cache "
                    f"version {active_version} is missing or incompatible"
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
            active_version = 0
            expected_teacher = "frozen_source"
            version_metadata = self.dynamic_hard_cache.metadata(active_version)
        if self.start_iteration == 0:
            self.refresh_cycle = 0
            active_version = 0
        self.dynamic_hard_cache.activate_version(active_version)
        active_metadata = self.dynamic_hard_cache.metadata(active_version)
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
                teacher_iteration=iteration,
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
        self.coverage_mode = True
        self.dynamic_hard_cache_memory.clear()
        self.dynamic_hard_memory_case = None
        metadata = self.dynamic_hard_cache.metadata(self.refresh_cycle)
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
                "coverage_mode": self.coverage_mode,
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
                "css_high_attempts": self.high_selection_attempts,
                "css_high_accepts": self.high_selection_accepts,
                "css_high_rejects": self.high_selection_rejects,
                "css_high_acceptance_rate": (
                    self.high_selection_accepts / max(self.high_selection_attempts, 1)
                ),
                "liver_loss_means": {
                    name: self.liver_loss_totals[name] / max(self.liver_loss_counts[name], 1)
                    for name in self.liver_loss_totals
                },
            },
            self.output_dir / "training_summary.json",
        )

    def _trainer_checkpoint_state(self) -> dict:
        state = {
            "dynamic_hard_enabled": self.dynamic_hard_enabled,
            "refresh_cycle": self.refresh_cycle,
            "seen_cases_since_refresh": sorted(self.seen_cases_since_refresh),
            "coverage_mode": self.coverage_mode,
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
        if self.high_selection_mode != "none":
            state["high_selection_state"] = {
                "schema_version": 1,
                "mode": self.high_selection_mode,
                "config_fingerprint": self._high_selection_config_fingerprint(),
                "checkpoint_student_fingerprint": self._lora_fingerprint(),
                "cache_iteration": self.high_selection_cache_iteration,
                "cache_version": self.high_reference_bank.cache_version,
                "cache_model_fingerprint": self.high_reference_bank.model_fingerprint,
                "reference_bank": self.high_reference_bank.state_dict(),
                "rng_state": self.high_selection_rng.getstate(),
                "attempts": self.high_selection_attempts,
                "accepts": self.high_selection_accepts,
                "rejects": self.high_selection_rejects,
                "optimizer_step_count": self.optimizer_step_count,
            }
        return state

    def _lora_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for parameter in self.runtime.trainable_parameters:
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _high_selection_config_fingerprint(self) -> str:
        route_files = {}
        for key in ("routing_uncertainty_csv", "routing_reference_partition"):
            raw_path = self.config["method"].get(key)
            if raw_path:
                path = Path(raw_path).expanduser()
                route_files[key] = {
                    "path": str(path.resolve()),
                    "exists": path.is_file(),
                    "sha256": _file_sha256(path) if path.is_file() else None,
                }
        parameter_names = {
            id(parameter): name for name, parameter in self.model.named_parameters()
        }
        payload = {
            "mode": self.high_selection_mode,
            "config": self.high_selection_config,
            "training_case_ids": sorted(case.name for case in self.train_cases),
            "routing_config": {
                key: self.config["method"].get(key)
                for key in ("routing_mode", "routing_uncertainty_csv", "routing_reference_partition", "routing_high_cases")
            },
            "routing_files": route_files,
            "high_pseudo_label_config": {
                key: self.config["method"].get(key)
                for key in (
                    "probability_threshold", "random_walk_size", "random_walk_steps",
                    "random_walk_alpha", "affinity_temperature",
                    "random_walk_score_pooling", "preserve_raw_foreground",
                    "high_pseudo_target",
                    "min_pseudo_foreground_voxels",
                )
            },
            "tile_step_size": self.config["train"].get("tile_step_size"),
            "patch_size": self.patch_size,
            "prompt": self.config["model"].get("prompt"),
            "source_checkpoint": source_checkpoint_signature(self.config["model"]),
            "trainable_parameters": [
                (parameter_names.get(id(parameter), "<unnamed>"), tuple(parameter.shape))
                for parameter in self.runtime.trainable_parameters
            ],
            "high_loss": {
                key: self.config["train"].get(key)
                for key in ("high_bce_weight", "high_dice_weight", "high_foreground_weight", "balance_high_bce")
            },
            "learning_rate": self.config["train"].get("learning_rate"),
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _restore_high_selection_state(self, training_state: dict) -> None:
        if getattr(self, "high_selection_mode", "none") == "none":
            return
        state = training_state.get("high_selection_state")
        if state is None:
            # Explicit safe recovery for older checkpoints: never reuse an
            # untracked cache; the next High candidate rebuilds it.
            self.high_reference_bank = ReferenceBank()
            self.high_selection_refs = self.high_reference_bank.gradients
            self.high_selection_features = self.high_reference_bank.features
            self.high_selection_cache_iteration = None
            self.logger.warning(
                "Checkpoint has no High Selection cache state; invalidating cache "
                "and rebuilding it at the next High candidate"
            )
            return
        if int(state.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported High Selection checkpoint state version")
        if state.get("mode") != self.high_selection_mode:
            raise ValueError("Checkpoint High Selection mode does not match config")
        if state.get("config_fingerprint") != self._high_selection_config_fingerprint():
            raise ValueError("Checkpoint High Selection config/routing is incompatible")
        current_fingerprint = self._lora_fingerprint()
        if state.get("checkpoint_student_fingerprint") != current_fingerprint:
            raise ValueError("Checkpoint High Selection state does not match loaded Student parameters")
        self.high_reference_bank.load_state_dict(state["reference_bank"])
        if self.high_reference_bank.cache_version != int(state.get("cache_version", -1)):
            raise ValueError("High Selection cache version metadata mismatch")
        if self.high_reference_bank.model_fingerprint != state.get("cache_model_fingerprint"):
            raise ValueError("High Selection cache snapshot fingerprint mismatch")
        if self.high_reference_bank.generated_iteration != state.get("cache_iteration"):
            raise ValueError("High Selection cache generation iteration mismatch")
        cache_iteration = state.get("cache_iteration")
        if cache_iteration is not None and int(cache_iteration) > self.last_training_iteration:
            raise ValueError("High Selection cache was generated after checkpoint iteration")
        self.high_selection_cache_iteration = None if cache_iteration is None else int(cache_iteration)
        self.high_selection_refs = self.high_reference_bank.gradients
        self.high_selection_features = self.high_reference_bank.features
        self.high_selection_rng.setstate(state["rng_state"])
        self.high_selection_attempts = int(state.get("attempts", 0))
        self.high_selection_accepts = int(state.get("accepts", 0))
        self.high_selection_rejects = int(state.get("rejects", 0))
        self.optimizer_step_count = int(state.get("optimizer_step_count", 0))

    def _sample_case_index(
        self,
        high_indices: list[int],
        low_indices: list[int],
    ) -> int:
        probability = float(
            self.config["method"].get("high_quality_sampling_probability", 0.5)
        )
        if self.dynamic_hard_enabled:
            return sample_coverage_index(
                high_indices,
                low_indices,
                [case.name for case in self.train_cases],
                self.seen_cases_since_refresh,
                probability,
                self.generator,
                coverage_mode=self.coverage_mode,
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
        if hasattr(self.store, "sliding_window_with_geometry"):
            case_volume, case_slicers, case_geometry = self.store.sliding_window_with_geometry(
                case,
                self.tile_step_size,
            )
        else:
            case_volume, case_slicers = self.store.sliding_window(
                case, self.tile_step_size
            )
            case_geometry = {}
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
        step_loss_components: dict[str, list[float]] = {}
        patch_count = len(case_slicers)
        selected_high = should_select_high(high_confidence, self.high_selection_mode)
        scale_before = float(self.scaler.get_scale()) if selected_high else 1.0
        liver_context = None
        volume_gradient = None
        volume_normalization = None
        volume_gaussian = None
        volume_loss_value = 0.0
        volume_ratio = float("nan")
        liver_experiment = getattr(self, "liver_experiment", "baseline")
        if high_confidence and liver_experiment in {"E1", "E2", "E3", "E4", "E5"}:
            liver_context = self._liver_teacher_context(
                case, case_volume, case_slicers, case_geometry
            )
        if high_confidence and liver_experiment in {"E1", "E2"}:
            signal = self._global_volume_constraint_signal(
                case_volume,
                case_slicers,
                case_geometry,
                liver_context["prediction"],
            )
            volume_gradient = signal["gradient"]
            volume_normalization = signal["normalization"]
            volume_gaussian = signal["gaussian"]
            volume_loss_value = signal["loss"]
            volume_ratio = signal["ratio"]
            self.liver_loss_totals["volume_constraint_loss"] = (
                self.liver_loss_totals.get("volume_constraint_loss", 0.0)
                + volume_loss_value
            )
            self.liver_loss_counts["volume_constraint_loss"] = (
                self.liver_loss_counts.get("volume_constraint_loss", 0) + 1
            )
            if np.isfinite(volume_ratio):
                self.liver_loss_totals["student_teacher_volume_ratio"] = (
                    self.liver_loss_totals.get("student_teacher_volume_ratio", 0.0)
                    + volume_ratio
                )
                self.liver_loss_counts["student_teacher_volume_ratio"] = (
                    self.liver_loss_counts.get("student_teacher_volume_ratio", 0) + 1
                )
        volume_effective = volume_gradient is not None and volume_loss_value > 0.0
        for patch_index, case_slicer in enumerate(case_slicers):
            partner_slicer = partner_slicers[patch_index % len(partner_slicers)]
            image = self._patch(case_volume, case_slicer)
            partner_image = self._patch(partner_volume, partner_slicer)
            if high_confidence:
                if getattr(self, "liver_config", None) or str(
                    self.config.get("method", {}).get("high_pseudo_target", "source")
                ) == "prompt_consensus":
                    loss, patch_metrics = self._high_confidence_loss(
                        image, case, case_slicer, liver_context
                    )
                else:
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
            for component in (
                "high_bce_loss",
                "high_dice_loss",
                "remote_background_loss",
                "region_loss",
                "anchor_loss",
            ):
                if component in patch_metrics:
                    loss_value = float(patch_metrics[component])
                    step_loss_components.setdefault(component, []).append(loss_value)
                    self.liver_loss_totals[component] = (
                        self.liver_loss_totals.get(component, 0.0) + loss_value
                    )
                    self.liver_loss_counts[component] = (
                        self.liver_loss_counts.get(component, 0) + 1
                    )
        if (
            volume_gradient is not None
            and volume_normalization is not None
            and volume_gaussian is not None
            and volume_loss_value > 0.0
        ):
            was_training = self.model.training
            self.model.train()
            rng_after_base_cpu = torch.get_rng_state()
            rng_after_base_cuda = (
                torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda"
                else None
            )
            for case_slicer, rng_state in zip(
                case_slicers, signal["rng_states"], strict=True
            ):
                torch.set_rng_state(rng_state[0])
                if self.device.type == "cuda":
                    torch.cuda.set_rng_state(rng_state[1], self.device)
                image = self._patch(case_volume, case_slicer)
                with self._autocast():
                    student_logits = self.model(image, self.embedding)
                coefficient = patch_vjp_coefficient(
                    volume_gradient,
                    volume_normalization,
                    volume_gaussian,
                    case_slicer,
                )
                surrogate = (student_logits[0, 0].float() * coefficient).sum()
                self.scaler.scale(surrogate).backward()
            torch.set_rng_state(rng_after_base_cpu)
            if self.device.type == "cuda" and rng_after_base_cuda is not None:
                torch.cuda.set_rng_state(rng_after_base_cuda, self.device)
            self.model.train(was_training)
            self.liver_loss_totals["volume_constraint_updates"] = (
                self.liver_loss_totals.get("volume_constraint_updates", 0.0) + 1.0
            )
            self.liver_loss_counts["volume_constraint_updates"] = (
                self.liver_loss_counts.get("volume_constraint_updates", 0) + 1
            )
        del volume_gradient, volume_normalization, volume_gaussian
        self.scaler.unscale_(self.optimizer)
        gradient_denominator = max(effective_patch_count, 1)
        for parameter in self.runtime.trainable_parameters:
            if parameter.grad is not None:
                parameter.grad.div_(gradient_denominator)
        selection: dict = {"accepted": True, "reason": "mode_none"}
        high_overflow = False
        if high_confidence and self.high_selection_mode != "none":
            finite_before_selection = gradients_are_finite(self.runtime.trainable_parameters)
            if not finite_before_selection:
                high_overflow = True
                selection = self._decide_high_update(
                    case, None, invalid_reason="gradient_overflow"
                )
            else:
                candidate = flatten_gradients(tuple(
                    parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
                    for parameter in self.runtime.trainable_parameters
                ))
                try:
                    selection = self._decide_high_update(case, candidate)
                except Exception:
                    reset_scaler_after_rejection(self.scaler, scale_before)
                    self.optimizer.zero_grad(set_to_none=True)
                    raise
        max_norm = float(self.config["train"]["max_grad_norm"])
        step_executed = False
        if selected_high:
            step_executed = conditional_scaler_step(
                self.scaler,
                self.optimizer,
                self.runtime.trainable_parameters,
                accepted=bool(selection["accepted"]),
                effective=effective_patch_count > 0 or volume_effective,
                overflow=high_overflow,
                max_norm=max_norm,
                scale_before=scale_before,
            )
            self.optimizer_step_count += int(step_executed)
        else:
            # Historical mode=none and Low AMP path: keep the original sequence.
            torch.nn.utils.clip_grad_norm_(self.runtime.trainable_parameters, max_norm)
            if effective_patch_count > 0 or volume_effective:
                will_skip = scaler_will_skip_step(self.scaler, self.optimizer)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if not will_skip:
                    self.optimizer_step_count += 1
            else:
                self.scaler.update()
        if high_confidence and self.high_selection_mode != "none":
            self._write_high_selection_row(
                self.high_selection_pending_row,
                step_executed=step_executed,
                skip_reason=(
                    "gradient_overflow" if high_overflow else
                    "amp_or_clipping_skip" if bool(selection["accepted"]) and not step_executed else ""
                ),
            )
            self.high_selection_pending_row = None
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
            "volume_constraint_loss": volume_loss_value,
            "student_teacher_volume_ratio": volume_ratio,
        }
        for component, values in step_loss_components.items():
            metrics[component] = float(np.mean(values))
        if self.dynamic_hard_enabled:
            metrics["dynamic_hard_loss"] = dynamic_hard_loss_sum / max(
                patch_count, 1
            )
        if high_confidence and self.high_selection_mode != "none":
            metrics.update({
                "selection_mode": self.high_selection_mode,
                "accepted": int(bool(selection["accepted"])),
                "rejection_reason": str(selection.get("reason", "")),
                "high_attempt_count": self.high_selection_attempts,
                "high_accept_count": self.high_selection_accepts,
                "high_reject_count": self.high_selection_rejects,
                "low_step_count": self.branch_sample_counts.get("low", 0),
                "optimizer_step_count": self.optimizer_step_count,
            })
        return metrics

    def _selection_reference_cases(self, excluded: str) -> list[CaseRecord]:
        high_cases = [case for case in self.train_cases if self.partition.is_high_confidence(case.name)]
        pool = [case for case in high_cases if case.name != excluded]
        count = max(2, int(self.high_selection_config.get("reference_count", 8)))
        count = min(count, len(pool))
        if count < 2:
            return []
        self.high_selection_rng.shuffle(pool)
        return sorted(pool[:count], key=lambda case: case.name)

    def _refresh_high_selection_cache(self, iteration: int, excluded: str) -> None:
        mode = self.high_selection_mode
        refs = self._selection_reference_cases(excluded)
        self.high_selection_refs = {}
        self.high_selection_features = {}
        for reference in refs:
            if mode in {"gradient_agreement", "cross_case_validation", "semantic_update_validation"}:
                try:
                    self.high_selection_refs[reference.name] = self._high_case_gradient(reference)
                except (ValueError, FloatingPointError, RuntimeError):
                    # Invalid/empty reference data remains unavailable and can
                    # cause a safe rejection; it never implies unconditional use.
                    pass
            if mode in {"semantic_subspace", "semantic_update_validation"}:
                try:
                    self.high_selection_features[reference.name] = self._semantic_case_features(reference)
                except (ValueError, FloatingPointError, RuntimeError):
                    pass
        semantic_evidence = None
        semantic_error = None
        if mode in {"semantic_subspace", "semantic_update_validation"}:
            try:
                semantic_evidence = build_semantic_cache(
                    self.high_selection_features,
                    float(self.high_selection_config.get("spectral_energy", 0.9)),
                    float(self.high_selection_config.get("robust_mad_multiplier", 3.0)),
                )
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                semantic_error = str(error)
        fingerprint = self._lora_fingerprint()
        self.high_reference_bank.replace(
            gradients=self.high_selection_refs,
            features=self.high_selection_features,
            reference_case_ids=[reference.name for reference in refs],
            semantic_evidence=semantic_evidence,
            semantic_cache_error=semantic_error,
            generated_iteration=iteration,
            model_fingerprint=fingerprint,
        )
        self.high_selection_refs = self.high_reference_bank.gradients
        self.high_selection_features = self.high_reference_bank.features
        self.high_selection_cache_iteration = int(iteration)
        save_json({
            "mode": mode,
            "reference_case_ids": self.high_reference_bank.reference_case_ids,
            "generated_iteration": int(iteration),
            "source_checkpoint": source_checkpoint_signature(self.config["model"]),
            "adapted_student_lora_sha256": fingerprint,
            "cache_refresh_interval": int(self.high_selection_config.get("reference_refresh_interval", 100)),
            "cache_version": self.high_reference_bank.cache_version,
            "semantic_cache_error": semantic_error,
            "stale_reference_policy": "snapshot references are approximate until the next configured refresh",
            "config": self.high_selection_config,
        }, self.output_dir / "high_selection_cache_metadata.json")

    def _high_case_gradient(self, case: CaseRecord) -> torch.Tensor:
        """Full-case High gradient. autograd.grad does not populate parameter .grad."""
        volume, slicers = self.store.sliding_window(case, self.tile_step_size)
        total: list[torch.Tensor | None] = [None] * len(self.runtime.trainable_parameters)
        effective = 0
        reference_scale = float(self.scaler.get_scale())
        was_training = self.model.training
        normalization_buffers = []
        for module in self.model.modules():
            for name in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(module, name, None)
                if isinstance(value, torch.Tensor):
                    normalization_buffers.append((module, name, value.detach().clone()))
        self.model.train(was_training)
        try:
            for slicer in slicers:
                loss, metrics = self._high_confidence_loss(
                    self._patch(volume, slicer), case, slicer
                )
                if not bool(metrics["effective"]) or not bool(torch.isfinite(loss)):
                    continue
                # autograd.grad returns tensors without populating .grad, so
                # GradScaler.unscale_ cannot apply here: undo this temporary
                # scale exactly once on each returned reference gradient.
                grads = torch.autograd.grad(loss * reference_scale, self.runtime.trainable_parameters, allow_unused=True)
                for index, grad in enumerate(grads):
                    if grad is None:
                        continue
                    grad = grad.detach().float() / reference_scale
                    total[index] = grad.clone() if total[index] is None else total[index] + grad
                effective += 1
        finally:
            with torch.no_grad():
                for module, name, value in normalization_buffers:
                    getattr(module, name).copy_(value)
            self.model.train(was_training)
        if effective == 0:
            raise ValueError(f"reference case has no effective High pseudo-label patches: {case.name}")
        result = flatten_gradients(tuple(
            torch.zeros_like(parameter) if grad is None else grad / effective
            for parameter, grad in zip(self.runtime.trainable_parameters, total, strict=True)
        ))
        if result is None:
            raise ValueError(f"reference case produced a zero or non-finite gradient: {case.name}")
        return result.cpu()

    def _semantic_case_features(self, case: CaseRecord) -> np.ndarray:
        """Sample source-LoRA d5_pre_head tokens under teacher foreground."""
        volume, slicers = self.store.sliding_window(case, self.tile_step_size)
        limit = max(1, int(self.high_selection_config.get("tokens_per_case", 256)))
        chunks = []
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad(), lora_disabled(self.model):
                for slicer in slicers:
                    image = self._patch(volume, slicer)
                    seed = int(self.config["experiment"]["seed"]) + sum(case.name.encode("utf-8")) + len(chunks)
                    tokens = foreground_d5_tokens(
                        self.model,
                        image,
                        self.embedding,
                        float(self.config["method"]["probability_threshold"]),
                        limit,
                        seed,
                        self._autocast,
                    )
                    if tokens.size:
                        chunks.append(tokens)
        finally:
            self.model.train(was_training)
        if not chunks:
            raise ValueError(f"semantic feature extraction found no candidate foreground: {case.name}")
        values = np.concatenate(chunks, axis=0)
        seed = int(self.config["experiment"]["seed"]) + sum(case.name.encode("utf-8"))
        rng = np.random.default_rng(seed)
        if len(values) > limit:
            values = values[rng.choice(len(values), size=limit, replace=False)]
        return values.astype(np.float32, copy=False)

    def _decide_high_update(
        self,
        case: CaseRecord,
        candidate: torch.Tensor | None,
        invalid_reason: str = "zero_or_nonfinite_candidate_gradient",
    ) -> dict:
        self.high_selection_attempts += 1
        result: dict
        if candidate is None:
            result = {"accepted": False, "reason": invalid_reason}
        else:
            iteration = self.last_training_iteration + 1
            interval = max(1, int(self.high_selection_config.get("reference_refresh_interval", 100)))
            if self.high_selection_cache_iteration is None or iteration - self.high_selection_cache_iteration >= interval:
                self._refresh_high_selection_cache(iteration, case.name)
            refs = self.high_reference_bank.gradient_references(case.name, candidate.device)
            mode = self.high_selection_mode
            details: dict = {}
            accepted = False
            reason = "insufficient_reference_data"
            grad_ok = False
            update_ok = False
            semantic_ok = False
            if mode == "gradient_agreement":
                decision = gradient_agreement.decide(candidate, refs)
                accepted, reason = bool(decision["accepted"]), str(decision["reason"])
                details.update({key: value for key, value in decision.items() if key not in {"accepted", "reason"}})
            if mode in {"cross_case_validation", "semantic_update_validation"}:
                decision = cross_case_validation.decide(candidate, refs, float(self.config["train"]["learning_rate"]), float(self.high_selection_config.get("significance_level", 0.05)))
                update_ok = bool(decision["accepted"])
                details.update({key: value for key, value in decision.items() if key not in {"accepted", "reason"}})
                if mode == "cross_case_validation":
                    accepted, reason = update_ok, str(decision["reason"])
            if mode in {"semantic_subspace", "semantic_update_validation"}:
                try:
                    candidate_features = self._semantic_case_features(case)
                    evidence = self.high_reference_bank.semantic_for(case.name)
                    basis = evidence["basis"].numpy()
                    rank = int(evidence["rank"])
                    spectral = float(evidence["spectral_energy"])
                    threshold = float(evidence["threshold"])
                    residual = relative_subspace_error(candidate_features, basis)
                    semantic_ok = residual <= threshold
                    details.update(subspace_rank=rank, retained_spectral_energy=spectral, relative_reconstruction_error=residual, reference_case_count=int(evidence["reference_case_count"]), semantic_accepted=semantic_ok, semantic_threshold=threshold)
                    if mode == "semantic_subspace":
                        accepted, reason = semantic_ok, "within_leave_one_case_out_subspace_threshold" if semantic_ok else "semantic_residual_above_threshold"
                except (ValueError, np.linalg.LinAlgError, FloatingPointError, RuntimeError) as error:
                    details.update(semantic_accepted=False, semantic_error=str(error))
                    if mode == "semantic_subspace":
                        reason = "semantic_subspace_unavailable"
            if mode == "semantic_update_validation":
                decision = semantic_update_validation.decide(semantic_ok, update_ok)
                accepted, reason = bool(decision["accepted"]), str(decision["reason"])
                details.update({key: value for key, value in decision.items() if key not in {"accepted", "reason"}})
            result = {"accepted": accepted, "reason": reason, **details}
        self.high_selection_accepts += int(bool(result["accepted"]))
        self.high_selection_rejects += int(not bool(result["accepted"]))
        row = {"iteration": self.last_training_iteration + 1, "candidate_case": case.name,
               "selection_mode": self.high_selection_mode, "accepted": int(bool(result["accepted"])),
               "rejection_reason": "" if result["accepted"] else result["reason"],
               "high_attempt_count": self.high_selection_attempts,
               "high_accept_count": self.high_selection_accepts,
               "high_reject_count": self.high_selection_rejects,
               "low_step_count": self.branch_sample_counts.get("low", 0),
               "optimizer_step_count": self.optimizer_step_count,
               "optimizer_step_executed": 0,
               "update_skip_reason": "",
               "reference_refresh_iteration": self.high_selection_cache_iteration, **result}
        self.high_selection_pending_row = row
        return result

    def _write_high_selection_row(
        self, row: dict | None, *, step_executed: bool, skip_reason: str
    ) -> None:
        if row is None:
            return
        row["optimizer_step_count"] = self.optimizer_step_count
        row["optimizer_step_executed"] = int(step_executed)
        row["update_skip_reason"] = skip_reason
        import csv
        path = self.output_dir / "high_selection_training_log.csv"
        fields = [
            "iteration", "candidate_case", "selection_mode", "accepted", "rejection_reason",
            "high_attempt_count", "high_accept_count", "high_reject_count", "low_step_count",
            "optimizer_step_count", "optimizer_step_executed", "update_skip_reason",
            "reference_refresh_iteration", "reason",
            "gradient_agreement_mean", "gradient_agreement_std", "reference_count",
            "estimated_delta_mean", "estimated_delta_std", "estimated_delta_upper_bound",
            "subspace_rank", "retained_spectral_energy", "relative_reconstruction_error",
            "reference_case_count", "semantic_accepted", "update_accepted", "final_accepted",
            "semantic_threshold", "semantic_error",
        ]
        write_header = not path.is_file() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

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

    def _liver_teacher_context(
        self,
        case: CaseRecord,
        volume: torch.Tensor,
        slicers: list[tuple[slice, slice, slice]],
        geometry: dict[str, object],
    ) -> dict:
        if self.source_volume_cache is None:
            raise RuntimeError("Liver adaptation requires a source-volume cache")
        if self.liver_context_case_name == case.name and self.liver_context_memory is not None:
            return self.liver_context_memory
        prediction = self.source_volume_cache.get_or_build(
            case,
            volume,
            slicers,
            geometry,
            self.store,
            self.runtime,
            self.embedding,
            self.config["model"],
            self.tile_step_size,
            self.logger,
        )
        teacher_probability = prediction.padded_probability(self.device)
        context: dict = {
            "valid": prediction.soft_volume >= float(
                self.liver_config.get("volume_constraint", {}).get(
                    "minimum_teacher_volume", 1.0
                )
            ),
            "prediction": prediction,
            "teacher_probability": teacher_probability,
        }
        if self.liver_experiment in {"E3", "E4"}:
            regions_config = self.liver_config.get("regions", {})
            remote_config = self.liver_config.get("remote_background", {})
            masks = build_liver_region_masks(
                prediction.probability,
                prediction.spacing,
                foreground_threshold=float(
                    (regions_config if self.liver_experiment == "E4" else remote_config)
                    .get("foreground_threshold", 0.5)
                ),
                remote_probability_max=float(remote_config.get("probability_max", 0.1)),
                remote_distance_mm=float(remote_config.get("distance_mm", 15.0)),
                boundary_width_mm=float(regions_config.get("boundary_width_mm", 5.0)),
            )
            context["valid"] = bool(masks["valid"].item())
            if context["valid"]:
                context["remote_background"] = prediction.padded_mask(
                    masks["remote_background"], self.device
                )
                if self.liver_experiment == "E4":
                    context["interior"] = prediction.padded_mask(
                        masks["interior"], self.device
                    )
                    context["boundary"] = prediction.padded_mask(
                        masks["boundary"], self.device
                    )
            else:
                self.logger.warning(
                    "Skipping Liver distance regions for Teacher-empty case=%s",
                    case.name,
                )
        self.liver_context_case_name = case.name
        self.liver_context_memory = context
        return context

    def _global_volume_constraint_signal(
        self,
        volume: torch.Tensor,
        slicers: list[tuple[slice, slice, slice]],
        geometry: dict[str, object],
        teacher_prediction: SourceVolumePrediction,
    ) -> dict:
        """Compute a whole-ROI loss and its exact tile-wise VJP coefficients."""
        shape = tuple(int(value) for value in geometry["padded_shape"])
        roi_slicer = geometry["roi_slicer"]
        patch_size = self.patch_size
        gaussian = compute_gaussian(
            patch_size,
            sigma_scale=1.0 / 8.0,
            value_scaling_factor=10,
            dtype=torch.float32,
            device=self.device,
        )
        accumulation = torch.zeros(shape, dtype=torch.float32, device=self.device)
        normalization = torch.zeros_like(accumulation)
        rng_states: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        rng_before_cpu = torch.get_rng_state()
        rng_before_cuda = (
            torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None
        )
        was_training = self.model.training
        self.model.train()
        try:
            with torch.no_grad():
                for slicer in slicers:
                    image = self._patch(volume, slicer)
                    cpu_rng = torch.get_rng_state()
                    cuda_rng = (
                        torch.cuda.get_rng_state(self.device)
                        if self.device.type == "cuda"
                        else None
                    )
                    rng_states.append((cpu_rng, cuda_rng))
                    with self._autocast():
                        logits = self.model(image, self.embedding)
                    accumulation[slicer] += logits[0, 0].float() * gaussian
                    normalization[slicer] += gaussian
            fused = accumulation / normalization.clamp_min(1e-6)
        finally:
            self.model.train(was_training)
        volume_config = self.liver_config.get("volume_constraint", {})
        loss_value, gradient, ratio, valid = fused_volume_constraint_gradient(
            fused,
            teacher_prediction.probability,
            roi_slicer,
            experiment=self.liver_experiment,
            margin=float(volume_config.get("margin", 0.1)),
            weight=float(volume_config.get("weight", 1.0)),
            shrink_weight=float(volume_config.get("shrink_weight", 1.0)),
            epsilon=float(volume_config.get("epsilon", 1e-6)),
            minimum_teacher_volume=float(volume_config.get("minimum_teacher_volume", 1.0)),
        )
        if gradient is None:
            torch.set_rng_state(rng_before_cpu)
            if self.device.type == "cuda" and rng_before_cuda is not None:
                torch.cuda.set_rng_state(rng_before_cuda, self.device)
        del fused, accumulation
        return {
            "gradient": gradient,
            "normalization": normalization if gradient is not None else None,
            "gaussian": gaussian if gradient is not None else None,
            "rng_states": rng_states,
            "loss": loss_value,
            "ratio": ratio,
        }

    def _high_confidence_loss(
        self,
        image: torch.Tensor,
        case: CaseRecord | None = None,
        case_slicer: tuple[slice, slice, slice] | None = None,
        liver_context: dict | None = None,
    ) -> tuple[torch.Tensor, dict[str, int | bool]]:
        method_config = self.config["method"]
        was_training = self.model.training
        teacher_image = image
        self.model.eval()
        with torch.no_grad():
            need_random_walk = self.liver_experiment != "E0"
            with lora_disabled(self.model):
                with self._autocast():
                    teacher_result = self.model(
                        teacher_image,
                        self.embedding,
                        return_features=need_random_walk,
                    )
            if need_random_walk:
                teacher_logits, features = teacher_result
            else:
                teacher_logits = teacher_result
                features = None
            source_probability = torch.sigmoid(teacher_logits.float())
            threshold = float(method_config["probability_threshold"])
            confidence = None
            if str(method_config.get("high_pseudo_target", "source")) == "prompt_consensus":
                if case is None or case_slicer is None:
                    raise ValueError("prompt-consensus High targets require case geometry")
                teacher_probability, confidence = self._dynamic_hard_patch(case, case_slicer)
            else:
                teacher_probability = source_probability
            raw_voxels = int((teacher_probability >= threshold).sum().item())
            if self.liver_experiment == "E0":
                # E0 is deliberately the fixed-source soft target ablation.
                refined = teacher_probability
            else:
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
                    refined = torch.maximum(refined, teacher_probability)
                del propagated
            del features
            del teacher_logits
            del teacher_probability
        self.model.train(was_training)
        with self._autocast():
            student_logits = self.model(teacher_image, self.embedding)
        self.model.train(was_training)
        student_probability = torch.sigmoid(student_logits.float())
        # Once a case passes Eq. (11), the paper supervises the complete
        # affinity-refined pseudo-label rather than masking it a second time.
        weight = torch.ones_like(refined) if confidence is None else confidence
        pseudo_voxels = int((refined >= threshold).sum().item())
        minimum = int(method_config.get("min_pseudo_foreground_voxels", 8))
        base_effective = pseudo_voxels >= minimum
        patch_metrics = {
            "raw_foreground_voxels": raw_voxels,
            "refined_foreground_voxels": pseudo_voxels,
            "voxels": refined.numel(),
            "effective": base_effective,
        }
        if base_effective and bool(self.config["train"].get("balance_high_bce", True)):
            bce = balanced_masked_binary_cross_entropy(student_probability, refined, weight)
            dice = masked_soft_dice_loss(student_probability, refined, weight)
            train_config = self.config["train"]
            loss = (
                float(train_config["high_bce_weight"]) * bce
                + float(train_config["high_dice_weight"]) * dice
            )
            patch_metrics["high_bce_loss"] = float(bce.detach())
            patch_metrics["high_dice_loss"] = float(dice.detach())
        elif base_effective:
            foreground_weight = float(
                self.config["train"].get("high_foreground_weight", 5.0)
            )
            weighted_voxels = weight * (
                1.0 + (foreground_weight - 1.0) * refined.detach()
            )
            bce = masked_binary_cross_entropy(student_probability, refined, weighted_voxels)
            dice = masked_soft_dice_loss(student_probability, refined, weighted_voxels)
            train_config = self.config["train"]
            loss = (
                float(train_config["high_bce_weight"]) * bce
                + float(train_config["high_dice_weight"]) * dice
            )
            patch_metrics["high_bce_loss"] = float(bce.detach())
            patch_metrics["high_dice_loss"] = float(dice.detach())
        else:
            bce = student_logits.float().sum() * 0.0
            dice = bce
            loss = bce
            patch_metrics["high_bce_loss"] = 0.0
            patch_metrics["high_dice_loss"] = 0.0
        extra_components: dict[str, float] = {}
        extra_effective = False
        if liver_context is not None and bool(liver_context.get("valid", False)):
            if self.liver_experiment == "E3":
                remote = liver_context["remote_background"][
                    (slice(None), slice(None), *case_slicer)
                ]
                remote_loss = remote_background_loss(student_probability, remote)
                weighted = float(
                    self.liver_config.get("remote_background", {}).get("weight", 0.1)
                ) * remote_loss
                loss = loss + weighted
                extra_components["remote_background_loss"] = float(remote_loss.detach())
                extra_effective = bool(remote.any()) and bool(
                    (student_probability.detach()[remote] > 1e-6).any()
                )
            elif self.liver_experiment == "E4":
                idx = (slice(None), slice(None), *case_slicer)
                region_loss = partitioned_teacher_loss(
                    liver_context["teacher_probability"][idx],
                    student_probability,
                    liver_context["interior"][idx],
                    liver_context["boundary"][idx],
                    liver_context["remote_background"][idx],
                    interior_weight=float(self.liver_config.get("regions", {}).get("interior_weight", 1.0)),
                    boundary_weight=float(self.liver_config.get("regions", {}).get("boundary_weight", 0.25)),
                    background_weight=float(self.liver_config.get("regions", {}).get("background_weight", 1.0)),
                )
                weighted = float(
                    self.liver_config.get("regions", {}).get("loss_weight", 0.1)
                ) * region_loss
                loss = loss + weighted
                extra_components["region_loss"] = float(region_loss.detach())
                region_mask = (
                    liver_context["interior"][idx]
                    | liver_context["boundary"][idx]
                    | liver_context["remote_background"][idx]
                )
                extra_effective = bool(region_mask.any()) and bool(
                    (
                        student_probability.detach()[region_mask]
                        - liver_context["teacher_probability"][idx][region_mask]
                    ).abs().gt(1e-6).any()
                )
            elif self.liver_experiment == "E5":
                idx = (slice(None), slice(None), *case_slicer)
                anchor_loss = teacher_anchor_loss(
                    liver_context["teacher_probability"][idx],
                    student_probability,
                    foreground_threshold=float(self.liver_config.get("anchor", {}).get("foreground_threshold", 0.9)),
                    background_threshold=float(self.liver_config.get("anchor", {}).get("background_threshold", 0.1)),
                    boundary_weight=float(self.liver_config.get("anchor", {}).get("boundary_weight", 0.1)),
                )
                weighted = float(self.liver_config.get("anchor", {}).get("weight", 0.1)) * anchor_loss
                loss = loss + weighted
                extra_components["anchor_loss"] = float(anchor_loss.detach())
                teacher_patch = liver_context["teacher_probability"][idx]
                extra_effective = bool(
                    (student_probability.detach() - teacher_patch).abs().gt(1e-6).any()
                )
        patch_metrics["effective"] = base_effective or extra_effective
        patch_metrics.update(extra_components)
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
            dynamic_hard_loss_tensor = dynamic_hard_consensus_loss(
                student_probability,
                target,
                confidence,
                float(self.dynamic_hard_config.get("confidence_threshold", 0.0)),
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
        if self.dynamic_hard_enabled:
            self.logger.info(
                "update_counters iteration=%d high_samples=%d high_effective_patches=%d "
                "low_samples=%d low_effective_patches=%d css_high_accepts=%d/%d "
                "css_acceptance_rate=%.4f",
                iteration,
                self.branch_sample_counts["high"],
                self.branch_effective_patch_counts["high"],
                self.branch_sample_counts["low"],
                self.branch_effective_patch_counts["low"],
                self.high_selection_accepts,
                self.high_selection_attempts,
                self.high_selection_accepts / max(self.high_selection_attempts, 1),
            )
        self._write_early_stopping()
        return should_stop

    def _autocast(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _save(self, path: Path, iteration: int) -> Path:
        trainer_state = (
            self._trainer_checkpoint_state()
            if self.dynamic_hard_enabled
            else None
        )
        if trainer_state is not None and self.high_selection_mode != "none":
            selection_state = trainer_state.pop("high_selection_state")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_sidecar = path.with_name(path.name + ".high_selection.tmp")
            torch.save(selection_state, temporary_sidecar)
            sidecar_digest = _file_sha256(temporary_sidecar)
            sidecar_name = f"{path.name}.high_selection.{sidecar_digest[:20]}.pt"
            sidecar_path = path.with_name(sidecar_name)
            if sidecar_path.exists():
                temporary_sidecar.unlink()
            else:
                temporary_sidecar.replace(sidecar_path)
            trainer_state["high_selection_sidecar"] = {
                "file": sidecar_name,
                "sha256": sidecar_digest,
                "cache_version": int(selection_state["cache_version"]),
                "checkpoint_student_fingerprint": selection_state["checkpoint_student_fingerprint"],
            }
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
            trainer_state=trainer_state,
        )
        if trainer_state is not None and "high_selection_sidecar" in trainer_state:
            current_sidecar = trainer_state["high_selection_sidecar"]["file"]
            for old_sidecar in path.parent.glob(path.name + ".high_selection.*.pt"):
                if old_sidecar.name != current_sidecar:
                    old_sidecar.unlink()
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
                for sidecar in previous_path.parent.glob(previous_path.name + ".high_selection.*.pt"):
                    sidecar.unlink()
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
        sidecar_reference = training_state.get("high_selection_sidecar")
        if sidecar_reference is not None and self.high_selection_mode != "none":
            name = str(sidecar_reference.get("file", ""))
            if not name or Path(name).name != name:
                raise ValueError("Invalid High Selection checkpoint sidecar path")
            sidecar_path = Path(path).expanduser().resolve().parent / name
            if not sidecar_path.is_file():
                raise FileNotFoundError(f"High Selection checkpoint sidecar is missing: {sidecar_path}")
            digest = _file_sha256(sidecar_path)
            if digest != sidecar_reference.get("sha256"):
                raise ValueError("High Selection checkpoint sidecar checksum mismatch")
            selection_state = torch.load(sidecar_path, map_location="cpu", weights_only=False)
            if (
                int(selection_state.get("cache_version", -1)) != int(sidecar_reference.get("cache_version", -2))
                or selection_state.get("checkpoint_student_fingerprint")
                != sidecar_reference.get("checkpoint_student_fingerprint")
            ):
                raise ValueError("High Selection sidecar binding metadata mismatch")
            training_state["high_selection_state"] = selection_state
        if self.dynamic_hard_enabled and not training_state.get(
            "dynamic_hard_enabled", False
        ):
            raise ValueError(
                "Cannot resume M1 Dynamic-hard from a checkpoint without its "
                "Dynamic-hard trainer state"
            )
        self.refresh_cycle = int(training_state.get("refresh_cycle", 0))
        self.seen_cases_since_refresh = set(
            str(name) for name in training_state.get("seen_cases_since_refresh", [])
        )
        self.coverage_mode = bool(training_state.get("coverage_mode", False))
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
        self._restore_high_selection_state(training_state)
        self.logger.info("Resumed from %s at iteration %d", path, self.start_iteration)
