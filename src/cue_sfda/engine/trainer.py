from __future__ import annotations

from contextlib import nullcontext
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from cue_sfda.data import CaseRecord, TargetVolumeStore
from cue_sfda.engine.checkpoint import load_checkpoint, save_checkpoint
from cue_sfda.evaluation import evaluate
from cue_sfda.method.augmentations import StrongViewGenerator
from cue_sfda.method.losses import (
    balanced_masked_binary_cross_entropy,
    bernoulli_kl_divergence,
    binary_entropy,
    confidence_weight_map,
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
from cue_sfda.modeling.voxtell import VoxTellRuntime
from cue_sfda.modeling.lora import lora_disabled
from cue_sfda.utils import save_json
from cue_sfda.sat_rescue import evaluate_sat_assisted
from cue_sfda.method.retrieval import StableSupportMemory
from cue_sfda.method.consensus import ConsensusCache, ConsensusCacheBuilder
from cue_sfda.modeling.voxtell import load_prompt_embedding
from cue_sfda.method.high_pseudo_cache import HighPseudoCache


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
        high_cache_dir = config.get("train", {}).get("high_pseudo_cache_dir")
        high_cache_cases = config.get("train", {}).get("high_pseudo_cache_case_ids")
        self.high_pseudo_cache = (
            HighPseudoCache(
                high_cache_dir,
                expected_cases=(high_cache_cases or [case.name for case in train_cases]),
            )
            if high_cache_dir else None
        )
        high_weighting = train_config.get("high_voxel_weighting", {})
        self.high_voxel_weighting = dict(high_weighting) if isinstance(high_weighting, dict) else {}
        self.high_voxel_weighting_enabled = bool(self.high_voxel_weighting.get("enabled", False))
        self.high_confidence_cache = None
        self.high_confidence_case_memory: dict | None = None
        self.high_confidence_memory_case: str | None = None
        if self.high_voxel_weighting_enabled:
            confidence_cache_dir = self.high_voxel_weighting.get("cache_dir")
            if not confidence_cache_dir:
                raise ValueError("high voxel weighting requires train.high_voxel_weighting.cache_dir")
            self.high_confidence_cache = ConsensusCache(confidence_cache_dir)
        relation_config = config.get("high_relation", {})
        self.high_relation_enabled = bool(relation_config.get("enabled", False))
        self.high_relation_weight = float(relation_config.get("weight", 0.0))
        self.high_relation_top_k = max(1, int(relation_config.get("top_k", 5)))
        self.high_relation_temperature = max(1e-6, float(relation_config.get("temperature", 0.1)))
        if self.high_relation_enabled and self.high_pseudo_cache is None:
            raise ValueError("high_relation requires train.high_pseudo_cache_dir")
        low_relation_config = config.get("low_relation", {})
        self.low_relation_enabled = bool(low_relation_config.get("enabled", False))
        self.low_relation_weight = float(low_relation_config.get("weight", 0.0))
        self.low_relation_checkpointing = bool(low_relation_config.get("gradient_checkpointing", False))
        if self.low_relation_enabled and self.high_pseudo_cache is None:
            raise ValueError("low_relation requires train.high_pseudo_cache_dir")
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
        routing_config = config.get("optimizer_routing", {})
        self.optimizer_routing_enabled = bool(routing_config.get("enabled", False))
        self.optimizer_routing_mode = str(routing_config.get("mode", "cosine"))
        self.optimizer_routing_fraction = float(routing_config.get("group_fraction", 0.5))
        if self.optimizer_routing_enabled and self.high_pseudo_cache is None:
            raise ValueError("optimizer_routing requires the frozen High pseudo-label cache")
        if not 0.0 < self.optimizer_routing_fraction <= 1.0:
            raise ValueError("optimizer_routing.group_fraction must be in (0, 1]")
        if self.optimizer_routing_mode not in {"fixed", "random", "norm", "cosine"}:
            raise ValueError("optimizer_routing.mode must be fixed/random/norm/cosine")
        self.route_generator = torch.Generator().manual_seed(seed + 100_000)
        self.parameter_groups: dict[str, list[torch.nn.Parameter]] = {}
        if self.optimizer_routing_enabled:
            trainable_ids = {id(parameter) for parameter in runtime.trainable_parameters}
            for name, parameter in self.model.named_parameters():
                if id(parameter) not in trainable_ids:
                    continue
                if ".layers." not in name or ".lora_" not in name:
                    continue
                layer = name.split(".layers.", 1)[1].split(".", 1)[0]
                projection = name.split(".lora_", 1)[1][0].upper()
                group_name = f"layer{layer}_{projection}"
                self.parameter_groups.setdefault(group_name, []).append(parameter)
            if not self.parameter_groups:
                raise RuntimeError("No decoder layer × Q/K/V parameter groups found")
        self.parameter_route_rows: list[dict[str, object]] = []
        self.high_reference_specs: list[tuple[CaseRecord, int]] = []
        self.start_iteration = 0
        self.best_dice = float("-inf")
        self.best_sat_dice = float("-inf")
        stage2 = config.get("stage2", {})
        self.stage2_mode = str(stage2.get("mode", "none"))
        prototype_strategy = str(stage2.get("prototype_strategy", ""))
        if not prototype_strategy:
            prototype_strategy = "spatial" if self.stage2_mode in {"prior_fixed", "prior", "weighted_prior"} else "clustered"
        self.stage2_memory = StableSupportMemory(
            max_local=int(stage2.get("prototype_count", 4)),
            momentum=float(stage2.get("memory_momentum", 0.9)),
            clustered=prototype_strategy == "clustered",
        )
        self.stage2_top_k = int(stage2.get("top_k", 3))
        self.stage2_temperature = float(stage2.get("similarity_temperature", 0.15))
        self.stage2_proto_weight = float(stage2.get("lambda_proto", 0.0))
        if self.stage2_mode in {"prior_fixed", "prior", "global", "local", "full"} and self.stage2_proto_weight <= 0.0:
            self.stage2_proto_weight = 0.02
        self.stage2_prior_weight = float(stage2.get("lambda_prior", 0.05))
        self.stage2_boundary_weight = float(stage2.get("lambda_boundary", 0.01))
        self.stage2_uncertain_low = float(stage2.get("uncertain_low", 0.2))
        self.stage2_uncertain_high = float(stage2.get("uncertain_high", 0.8))
        self.stage2_refine = bool(stage2.get("query_refinement", False))
        # The old foreground-only attraction is retained solely for the
        # explicitly named reproduction ablation. New prior variants never
        # pull every uncertain voxel toward foreground.
        self.stage2_legacy_proto = bool(stage2.get("legacy_proto_loss", False)) or self.stage2_mode in {"prior_fixed", "prior", "global", "local", "full"}
        self.best_iteration: int | None = None
        self.best_checkpoint_path: Path | None = None
        self.last_validation_iteration: int | None = None
        self.patience_counter = 0
        self.stop_iteration: int | None = None
        self.stop_reason: str | None = None
        self.calibration_foreground_counts: dict[str, int] = {}
        self.skipped_case_names: set[str] = set()
        self.routing_metadata: dict[str, int | float | str] = {}
        consensus = config.get("consensus", {})
        self.consensus_mode = str(consensus.get("mode", "none"))
        self.consensus_enabled = self.consensus_mode != "none"
        if self.consensus_mode in {"dynamic_hard_language", "selective_dynamic_language"}:
            consensus["language_verification"] = True
        self.consensus_cache: ConsensusCache | None = None
        self.consensus_case_memory: dict[str, dict] = {}
        self.consensus_memory_case: str | None = None
        self.refresh_cycle = 0
        self.max_refresh_cycles = int(consensus.get("max_refresh_cycles", 0))
        self.min_refresh_interval = int(consensus.get("min_refresh_interval", 0))
        self.seen_cases_since_refresh: set[str] = set()
        self.refresh_history: list[dict] = []
        self.consensus_prompt_embeddings: dict[str, torch.Tensor] = {}
        self.consensus_refresh_in_progress = False
        if self.consensus_enabled:
            cache_dir = consensus.get("cache_dir") or str(self.output_dir / "consensus_cache")
            self.consensus_cache = ConsensusCache(cache_dir)
            standard_prompt = str(consensus.get("standard_prompt", config["model"]["prompt"]))
            self.consensus_prompt_embeddings[standard_prompt] = self.embedding
            for prompt in consensus.get("prompts", config["method"].get("routing_prompts", [])):
                prompt = str(prompt)
                if prompt not in self.consensus_prompt_embeddings:
                    self.consensus_prompt_embeddings[prompt] = load_prompt_embedding(
                        config, prompt, self.output_dir / "prompt_cache", self.device
                    )
        rescue = config.get("sat_rescue", {})
        self.sat_proposal_dir = Path(rescue["proposal_dir"]).expanduser() if rescue.get("proposal_dir") else None
        self.sat_gate_csv = Path(rescue["gate_csv"]).expanduser() if rescue.get("gate_csv") else None
        self.sat_gate_threshold = float(rescue.get("gate_threshold", 0.01))
        self.sat_gate_mode = str(rescue.get("gate_mode", "response"))
        self.sat_rescue_cases: set[str] = set()
        self.sat_proposals: dict[str, np.ndarray] = {}
        if self.sat_proposal_dir and self.sat_gate_csv and self.sat_gate_csv.is_file():
            with self.sat_gate_csv.open(encoding="utf-8-sig", newline="") as handle:
                import csv
                for row in csv.DictReader(handle):
                    volume_ok = int(float(row.get("sat_prediction_voxels", 0))) > 0
                    response_ok = float(row.get("voxtell_max_probability", 0.0)) > self.sat_gate_threshold
                    gate_ok = volume_ok if self.sat_gate_mode in {"none", "volume"} else (volume_ok and response_ok)
                    if gate_ok:
                        self.sat_rescue_cases.add(str(row["case_id"]))
            self.logger.info("SAT rescue fixed train gate: mode=%s threshold=%.4f selected=%d", self.sat_gate_mode, self.sat_gate_threshold, len(self.sat_rescue_cases))

    def run(
        self,
        resume: str | Path | None = None,
        confidence_partition: str | Path | None = None,
    ) -> Path:
        if resume is not None:
            self._resume(resume)
            # Rebuild fixed D routing metadata after resume. Checkpoints carry
            # confidence scores, but routing skip/rescue sets are derived state.
            if self.config["method"].get("routing_uncertainty_csv"):
                self._initialize_routing(str(self.config["method"].get("routing_mode", "original")))
        elif confidence_partition is not None:
            self._load_partition(confidence_partition)
        if self.consensus_enabled:
            self._ensure_consensus_cache()
        else:
            save_json(
                {"mode": "none", "cache_version": None, "routing_frozen": True},
                self.output_dir / "consensus_cache_metadata.json",
            )
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
        total_iterations = int(method_config.get("max_iterations", method_config["iterations"]))
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
        if self.optimizer_routing_enabled:
            minimum = int(self.config["method"].get("min_pseudo_foreground_voxels", 8))
            for index in high_indices:
                case = self.train_cases[index]
                patch_index = self.high_pseudo_cache.first_effective_patch(case.name, minimum)
                if patch_index is not None:
                    self.high_reference_specs.append((case, patch_index))
            if not self.high_reference_specs:
                raise RuntimeError("No effective frozen High patch is available for routing references")
            self.logger.info("QKV routing reference patches=%d groups=%d mode=%s fraction=%.2f", len(self.high_reference_specs), len(self.parameter_groups), self.optimizer_routing_mode, self.optimizer_routing_fraction)
        if self.stage2_mode != "none" and self.start_iteration == 0:
            self._prime_stage2_memory(high_indices)
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
            metrics = self._train_step(case, partner, iteration)
            # Refresh coverage is based on cases actually sampled, not on
            # whether their pseudo-label happened to contain an effective
            # patch.  Otherwise all-background/skipped cases could prevent a
            # dynamic teacher from ever refreshing.
            if self.consensus_enabled:
                self.seen_cases_since_refresh.add(case.name)
            if iteration == 1 or iteration % log_every == 0:
                self.logger.info(
                    "iteration=%d/%d case=%s patches=%d effective_patches=%d skipped_patches=%d branch=%s loss=%.6f relation_loss=%.6f route_groups=%s confidence=%.4f raw_foreground=%.6f refined_foreground=%.6f split=%.4f",
                    iteration,
                    total_iterations,
                    case.name,
                    metrics["patches"],
                    metrics["effective_patches"],
                    metrics["skipped_patches"],
                    metrics["branch"],
                    metrics["loss"],
                    metrics["relation_loss"],
                    metrics.get("route_groups", "all"),
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
                    break
            if self._refresh_ready():
                self._refresh_consensus(iteration)
        if self.stop_iteration is None and self.validation_cases and self.last_validation_iteration != total_iterations:
            should_stop = self._validate(total_iterations)
            if should_stop:
                self.stop_iteration = total_iterations
                self.stop_reason = "early_stopping_patience"
        if self.stop_iteration is None:
            self.stop_iteration = total_iterations
            self.stop_reason = "max_iterations"
        final_iteration = self.stop_iteration
        latest_path = self._save_latest(final_iteration)
        self.logger.info("Saved latest LoRA checkpoint: %s", latest_path)
        self._write_parameter_routing()
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
        )
        self.partition.initialize(scores)
        self.skipped_case_names = {
            row.case_id for row in assignments if row.assigned_branch == "skip"
        }
        self.sat_rescue_cases &= self.skipped_case_names
        self.skipped_case_names -= self.sat_rescue_cases
        if self.sat_proposal_dir:
            for name in self.sat_rescue_cases:
                p = self.sat_proposal_dir / f"{name}.npy"
                if p.is_file():
                    self.sat_proposals[name] = np.load(p).astype(np.float32).squeeze()
                else:
                    self.logger.warning("Missing SAT proposal for rescue case %s", name)
            self.sat_rescue_cases &= set(self.sat_proposals)
        self.routing_metadata["sat_rescue"] = len(self.sat_rescue_cases)
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

    def _ensure_consensus_cache(self) -> None:
        if self.consensus_cache is None:
            return
        expected = {case.name for case in self.train_cases}
        metadata = self.consensus_cache.read_metadata()
        cached = set(metadata.get("case_ids", []))
        expected_prompts = [str(v) for v in self.config.get("consensus", {}).get("prompts", [])]
        compatible = int(metadata.get("num_views", -1)) == int(self.config["method"]["num_views"]) and [str(v) for v in metadata.get("prompts", [])] == expected_prompts
        if compatible and cached == expected and all(self.consensus_cache.has_case(case.name) for case in self.train_cases):
            if self.start_iteration == 0 and not self.refresh_history:
                self.refresh_cycle = int(metadata.get("cache_version", 0))
            save_json({"mode": self.consensus_mode, **metadata}, self.output_dir / "consensus_cache_metadata.json")
            self.logger.info("Using consensus cache=%s version=%d cases=%d", self.consensus_cache.root, self.refresh_cycle, len(cached))
            return
        self.logger.info("Consensus cache is incomplete; creating frozen source cache version 0")
        builder = ConsensusCacheBuilder(
            self.config,
            self.runtime,
            self.store,
            StrongViewGenerator(self.config["method"], int(self.config["experiment"]["seed"]) + 10000),
            self.consensus_prompt_embeddings,
            self.logger,
        )
        builder.disable_lora = True
        builder.build(self.train_cases, self.consensus_cache, cycle=0, metadata_extra={"routing_frozen": True})
        if self.start_iteration == 0 and not self.refresh_history:
            self.refresh_cycle = 0
        save_json({"mode": self.consensus_mode, **self.consensus_cache.read_metadata()}, self.output_dir / "consensus_cache_metadata.json")
        self._write_refresh_history()

    def _refresh_ready(self) -> bool:
        if self.consensus_mode not in {
            "dynamic_hard",
            "dynamic_ema",
            "dynamic_hard_language",
            "selective_dynamic",
            "selective_dynamic_language",
        }:
            return False
        if self.max_refresh_cycles > 0 and self.refresh_cycle >= self.max_refresh_cycles:
            return False
        if self.consensus_refresh_in_progress:
            return False
        if self.refresh_history and self.min_refresh_interval > 0:
            previous = int(self.refresh_history[-1]["trigger_iteration"])
            if (self.last_validation_iteration or 0) - previous < self.min_refresh_interval:
                return False
        return self.seen_cases_since_refresh == {case.name for case in self.train_cases}

    def _refresh_consensus(self, iteration: int) -> None:
        if self.consensus_cache is None:
            return
        self.consensus_refresh_in_progress = True
        self.refresh_cycle += 1
        cycle = self.refresh_cycle
        before = self.refresh_history[-1]["trigger_iteration"] if self.refresh_history else 0
        teacher_cache = self.consensus_cache
        temporary_root = self.output_dir / f".consensus_refresh_{cycle:04d}"
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        selective = self.consensus_mode in {"selective_dynamic", "selective_dynamic_language"}
        language_verify = self.consensus_mode in {"dynamic_hard_language", "selective_dynamic_language"}
        target_cache = ConsensusCache(temporary_root)
        builder = ConsensusCacheBuilder(
            self.config,
            self.runtime,
            self.store,
            StrongViewGenerator(self.config["method"], int(self.config["experiment"]["seed"]) + 10000 + cycle),
            self.consensus_prompt_embeddings,
            self.logger,
        )
        builder.disable_lora = False
        was_training = self.model.training
        self.model.eval()
        try:
            original_verification = self.config.get("consensus", {}).get("language_verification", False)
            self.config.setdefault("consensus", {})["language_verification"] = language_verify
            try:
                builder.build(self.train_cases, target_cache, cycle=cycle, metadata_extra={"routing_frozen": True, "teacher_type": "student_snapshot"})
            finally:
                self.config["consensus"]["language_verification"] = original_verification
            if self.consensus_mode == "dynamic_ema":
                refresh_stats = self._cache_change_stats(teacher_cache, target_cache)
                beta = float(self.config.get("consensus", {}).get("cache_ema_beta", 0.7))
                self._merge_ema_cache(teacher_cache, target_cache, beta, cycle)
            elif selective:
                refresh_stats = self._merge_selective_cache(
                    teacher_cache,
                    target_cache,
                    cycle,
                    language_verify=language_verify,
                )
            else:
                if language_verify:
                    refresh_stats = self._filter_dynamic_language_expansion(teacher_cache, target_cache)
                else:
                    refresh_stats = self._cache_change_stats(teacher_cache, target_cache)
                self._replace_cache(teacher_cache, target_cache, cycle)
        finally:
            self.model.train(was_training)
            if temporary_root.exists():
                shutil.rmtree(temporary_root)
            self.consensus_refresh_in_progress = False
        covered = len(self.seen_cases_since_refresh)
        self.refresh_history.append(
            {
                "refresh_cycle": cycle,
                "trigger_iteration": int(iteration),
                "covered_cases": int(covered),
                "iterations_since_previous_refresh": int(iteration - before),
                "teacher_type": "student_snapshot",
                "cache_ema_beta": float(self.config.get("consensus", {}).get("cache_ema_beta", 0.7)) if self.consensus_mode == "dynamic_ema" else None,
                **(refresh_stats if "refresh_stats" in locals() else {}),
            }
        )
        self.seen_cases_since_refresh.clear()
        self.consensus_case_memory.clear()
        self.consensus_memory_case = None
        self._write_refresh_history()
        self.logger.info("consensus refresh complete cycle=%d iteration=%d", cycle, iteration)

    @staticmethod
    def _cache_fields() -> tuple[str, ...]:
        return (
            "image_consensus_probability",
            "prompt_consensus_probability",
            "image_voxel_consistency",
            "prompt_voxel_consistency",
            "joint_voxel_confidence",
            "joint_probability",
            "language_response_mean",
            "language_response_variance",
        )

    def _replace_cache(self, old: ConsensusCache, new: ConsensusCache, cycle: int) -> None:
        for case in self.train_cases:
            data = new.load_case(case)
            geometry = dict(data["geometry"])
            geometry.update(cache_model_version=int(cycle), refresh_cycle=int(cycle))
            old.save_case(
                case,
                {field: data[field] for field in self._cache_fields()},
                geometry,
                data["image_view_case_weights"],
                data["prompt_case_weights"],
            )
        old.write_metadata({**new.read_metadata(), "cache_version": cycle, "teacher_type": "student_snapshot"})

    def _cache_change_stats(self, old: ConsensusCache, new: ConsensusCache) -> dict[str, float]:
        total = added = removed = confidence_change = foreground_change = 0
        for case in self.train_cases:
            old_data = old.load_case(case)
            new_data = new.load_case(case)
            p_old = old_data["joint_probability"]
            p_new = new_data["joint_probability"]
            total += int(p_old.size)
            added += int(((p_new > 0.5) & (p_old <= 0.5)).sum())
            removed += int(((p_new <= 0.5) & (p_old > 0.5)).sum())
            foreground_change += int((p_new > 0.5).sum() - (p_old > 0.5).sum())
            confidence_change += float((new_data["joint_voxel_confidence"] - old_data["joint_voxel_confidence"]).mean())
        return {
            "accepted_voxel_ratio": 1.0,
            "rejected_voxel_ratio": 0.0,
            "dynamic_added_foreground_voxels": float(added),
            "dynamic_removed_foreground_voxels": float(removed),
            "mean_foreground_volume_change": float(foreground_change / max(len(self.train_cases), 1)),
            "mean_confidence_change": float(confidence_change / max(len(self.train_cases), 1)),
        }

    def _filter_dynamic_language_expansion(
        self,
        old: ConsensusCache,
        new: ConsensusCache,
    ) -> dict[str, float]:
        """Reject unsupported newly added foreground before hard cache replacement."""
        response_threshold = float(self.config.get("consensus", {}).get("language_response_threshold", 0.5))
        variance_threshold = float(self.config.get("consensus", {}).get("language_response_variance_threshold", 0.08))
        total = accepted = rejected = added = removed = foreground_change = confidence_change = 0
        for case in self.train_cases:
            old_data = old.load_case(case)
            new_data = new.load_case(case)
            p_old = old_data["joint_probability"]
            p_new = new_data["joint_probability"].copy()
            expansion = (p_new > 0.5) & (p_old <= 0.5)
            semantic_ok = (
                (new_data["language_response_mean"] >= response_threshold)
                & (new_data["language_response_variance"] <= variance_threshold)
            )
            rejected_mask = expansion & ~semantic_ok
            accepted_mask = expansion & semantic_ok
            # Unsupported proposals inherit the old teacher probability and
            # confidence, while the rest of the cache still uses hard refresh.
            p_new[rejected_mask] = p_old[rejected_mask]
            new_confidence = new_data["joint_voxel_confidence"].copy()
            new_confidence[rejected_mask] = old_data["joint_voxel_confidence"][rejected_mask]
            new_data["joint_probability"] = p_new
            new_data["joint_voxel_confidence"] = new_confidence
            geometry = dict(new_data["geometry"])
            new.save_case(
                case,
                {field: new_data[field] for field in self._cache_fields()},
                geometry,
                new_data["image_view_case_weights"],
                new_data["prompt_case_weights"],
            )
            total += int(expansion.size)
            accepted += int(accepted_mask.sum())
            rejected += int(rejected_mask.sum())
            added += int(accepted_mask.sum())
            removed += int(((p_new <= 0.5) & (p_old > 0.5)).sum())
            foreground_change += int((p_new > 0.5).sum() - (p_old > 0.5).sum())
            confidence_change += float((new_confidence - old_data["joint_voxel_confidence"]).mean())
        denominator = max(total, 1)
        return {
            "accepted_voxel_ratio": float(accepted / denominator),
            "rejected_voxel_ratio": float(rejected / denominator),
            "dynamic_added_foreground_voxels": float(added),
            "dynamic_removed_foreground_voxels": float(removed),
            "mean_foreground_volume_change": float(foreground_change / max(len(self.train_cases), 1)),
            "mean_confidence_change": float(confidence_change / max(len(self.train_cases), 1)),
        }

    def _merge_selective_cache(
        self,
        old: ConsensusCache,
        new: ConsensusCache,
        cycle: int,
        language_verify: bool,
    ) -> dict[str, float]:
        delta = float(self.config.get("consensus", {}).get("delta_conf", 0.02))
        response_threshold = float(self.config.get("consensus", {}).get("language_response_threshold", 0.5))
        variance_threshold = float(self.config.get("consensus", {}).get("language_response_variance_threshold", 0.08))
        accepted = rejected = expanded = 0
        total = foreground_change = confidence_change = 0
        for case in self.train_cases:
            old_data = old.load_case(case)
            new_data = new.load_case(case)
            p_old = old_data["joint_probability"]
            p_new = new_data["joint_probability"]
            c_old = old_data["joint_voxel_confidence"]
            c_new = new_data["joint_voxel_confidence"]
            disagreement = np.abs(p_new - p_old) > 1e-6
            take_new = c_new > (c_old + delta)
            added_foreground = (p_new > 0.5) & (p_old <= 0.5) & disagreement
            if language_verify:
                semantic_ok = (
                    (new_data["language_response_mean"] >= response_threshold)
                    & (new_data["language_response_variance"] <= variance_threshold)
                )
                take_new &= (~added_foreground) | semantic_ok
            merged = {}
            for field in self._cache_fields():
                merged[field] = np.where(take_new, new_data[field], old_data[field]).astype(np.float32)
            geometry = dict(old_data["geometry"])
            geometry.update(cache_model_version=int(cycle), refresh_cycle=int(cycle))
            old.save_case(case, merged, geometry, new_data["image_view_case_weights"], new_data["prompt_case_weights"])
            accepted += int(take_new.sum())
            total += int(take_new.size)
            rejected += int((disagreement & ~take_new).sum())
            expanded += int(added_foreground.sum())
            foreground_change += int(((p_new > 0.5).sum() - (p_old > 0.5).sum()))
            confidence_change += float((c_new - c_old).mean())
        old.write_metadata({**new.read_metadata(), "cache_version": cycle, "teacher_type": "selective_student_snapshot", "delta_conf": delta})
        denominator = max(total, 1)
        return {
            "accepted_voxel_ratio": float(accepted / denominator),
            "rejected_voxel_ratio": float(rejected / denominator),
            "mean_foreground_volume_change": float(foreground_change / max(len(self.train_cases), 1)),
            "mean_confidence_change": float(confidence_change / max(len(self.train_cases), 1)),
            "dynamic_added_foreground_voxels": float(expanded),
        }

    def _merge_ema_cache(self, old: ConsensusCache, new: ConsensusCache, beta: float, cycle: int) -> None:
        beta = float(np.clip(beta, 0.0, 1.0))
        for case in self.train_cases:
            old_data = old.load_case(case)
            new_data = new.load_case(case)
            merged = {}
            for field in ("image_consensus_probability", "prompt_consensus_probability", "image_voxel_consistency", "prompt_voxel_consistency", "joint_voxel_confidence", "joint_probability"):
                merged[field] = beta * old_data[field] + (1.0 - beta) * new_data[field]
            geometry = dict(old_data["geometry"])
            geometry["cache_model_version"] = int(cycle)
            geometry["refresh_cycle"] = int(cycle)
            old.save_case(case, merged, geometry, beta * old_data["image_view_case_weights"] + (1.0 - beta) * new_data["image_view_case_weights"], beta * old_data["prompt_case_weights"] + (1.0 - beta) * new_data["prompt_case_weights"])
        old.write_metadata({**new.read_metadata(), "cache_version": cycle, "teacher_type": "ema_student_snapshot", "cache_ema_beta": beta})

    def _write_refresh_history(self) -> None:
        save_json(
            {
                "cache_version": int(self.refresh_cycle),
                "refresh_history": self.refresh_history,
                "seen_cases_since_refresh": sorted(self.seen_cases_since_refresh),
            },
            self.output_dir / "refresh_history.json",
        )

    def _sample_case_index(
        self,
        high_indices: list[int],
        low_indices: list[int],
    ) -> int:
        # Coverage is an observation about the ordinary C_joint sampler.  It
        # must never alter that sampler, otherwise static consensus ablations
        # and dynamic refresh runs no longer share the same protocol.
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
        self, case: CaseRecord, partner: CaseRecord, iteration: int = 0
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
        high_reference_gradients = None
        if self.optimizer_routing_enabled and not high_confidence:
            high_reference_gradients = (
                self._compute_high_reference_gradients(iteration)
                if self.optimizer_routing_mode == "cosine" else {}
            )
        loss_sum = 0.0
        confidence_sum = 0.0
        raw_foreground_sum = 0
        refined_foreground_sum = 0
        relation_loss_sum = 0.0
        voxel_sum = 0
        effective_patch_count = 0
        skipped_patch_count = 0
        patch_count = len(case_slicers)
        for patch_index, case_slicer in enumerate(case_slicers):
            partner_slicer = partner_slicers[patch_index % len(partner_slicers)]
            image = self._patch(case_volume, case_slicer)
            raw_low_branch = bool(self.config["train"].get("low_raw_frozen_source", False))
            partner_image = (
                None if high_confidence or raw_low_branch
                else self._patch(partner_volume, partner_slicer)
            )
            if high_confidence:
                voxel_confidence = (
                    self._high_confidence_patch(case, case_slicer, case_volume.shape[-3:])
                    if self.high_voxel_weighting_enabled else None
                )
                loss, patch_metrics = self._high_confidence_loss(
                    image, case.name, patch_index, voxel_confidence
                )
                branch = "high"
            elif case.name in self.sat_rescue_cases:
                loss, patch_metrics = self._sat_rescue_loss(image, case, case_slicer, case_volume)
                branch = "sat_rescue"
            else:
                loss, patch_metrics = self._low_confidence_loss(image, partner_image, case, case_slicer)
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
            relation_loss_sum += float(patch_metrics.get("relation_loss", 0.0))
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
        route_groups = "all"
        if self.optimizer_routing_enabled and not high_confidence:
            route_groups = self._apply_optimizer_routing(high_reference_gradients, iteration)
        max_norm = float(self.config["train"]["max_grad_norm"])
        torch.nn.utils.clip_grad_norm_(self.runtime.trainable_parameters, max_norm)
        if effective_patch_count > 0:
            self.scaler.step(self.optimizer)
        self.scaler.update()
        case_confidence = confidence_sum / patch_count
        if bool(self.config["method"].get("update_confidence_partition", False)):
            self.partition.update(case.name, case_confidence)
        return {
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
            "relation_loss": relation_loss_sum / max(effective_patch_count, 1),
            "route_groups": route_groups,
        }

    def _prime_stage2_memory(self, high_indices: list[int]) -> None:
        """Populate stable support memory once before the first query update."""
        self.logger.info("Priming Stage-2 stable support memory: cases=%d", len(high_indices))
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for index in high_indices:
                case = self.train_cases[index]
                volume, slicers = self.store.sliding_window(case, self.tile_step_size)
                if not slicers:
                    continue
                image = self._patch(volume, slicers[0])
                with self._autocast():
                    logits, features = self.model(image, self.embedding, return_features=True)
                self.stage2_memory.update(case.name, features, torch.sigmoid(logits.float()))
        self.model.train(was_training)
        self.logger.info("Stage-2 stable support memory ready: entries=%d", len(self.stage2_memory.entries))

    def _sat_rescue_loss(self, image, case, case_slicer, case_volume):
        proposal = self.sat_proposals[case.name]
        # Use the exact same centered padding convention as TargetVolumeStore.
        from acvl_utils.cropping_and_padding.padding import pad_nd_image
        proposal_tensor = torch.from_numpy(proposal[None]).float()
        proposal_tensor, _ = pad_nd_image(proposal_tensor, self.patch_size, "constant", {"value": 0}, True, None)
        if tuple(proposal_tensor.shape[-3:]) != tuple(case_volume.shape[-3:]):
            shape = tuple(int(x) for x in case_volume.shape[-3:])
            fixed = torch.zeros((1, *shape), dtype=torch.float32)
            common = tuple(min(shape[i], proposal_tensor.shape[-3+i]) for i in range(3))
            fixed[:, :common[0], :common[1], :common[2]] = proposal_tensor[:, :common[0], :common[1], :common[2]]
            proposal_tensor = fixed
        patch_mask = proposal_tensor[(slice(None), *case_slicer)].to(self.device).unsqueeze(0)
        with self._autocast():
            logits = self.model(image, self.embedding)
            if bool(patch_mask.any()):
                loss = torch.nn.functional.softplus(-logits[patch_mask.bool()]).mean()
                effective = True
            else:
                loss = logits.sum() * 0.0
                effective = False
        vox = int(patch_mask.sum().item())
        return loss, {"raw_foreground_voxels": vox, "refined_foreground_voxels": vox,
                      "voxels": int(patch_mask.numel()), "effective": effective}

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
        case_name: str | None = None,
        patch_index: int | None = None,
        voxel_confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, int | bool]]:
        method_config = self.config["method"]
        teacher_image = image
        threshold = float(method_config["probability_threshold"])
        if self.high_pseudo_cache is not None:
            if case_name is None or patch_index is None:
                raise ValueError("Cached high supervision requires case name and patch index")
            cached, raw_voxels, pseudo_voxels = self.high_pseudo_cache.target(
                case_name, patch_index, self.device
            )
            refined = cached
            need_features = self.high_relation_enabled
            with self._autocast():
                if need_features:
                    student_logits, student_features = self.model(
                        teacher_image, self.embedding, return_features=True
                    )
                else:
                    student_logits = self.model(teacher_image, self.embedding)
                    student_features = None
        else:
            self.model.eval()
            with torch.no_grad():
                with lora_disabled(self.model):
                    with self._autocast():
                        teacher_logits, features = self.model(
                            teacher_image, self.embedding, return_features=True,
                        )
                teacher_probability = torch.sigmoid(teacher_logits.float())
                raw_voxels = int((teacher_probability >= threshold).sum().item())
                propagated = affinity_random_walk(
                    probability=teacher_probability, features=features, image=teacher_image,
                    output_size=tuple(teacher_probability.shape[-3:]),
                    graph_size=tuple(int(value) for value in method_config["random_walk_size"]),
                    steps=int(method_config["random_walk_steps"]),
                    alpha=float(method_config["random_walk_alpha"]),
                    temperature=float(method_config["affinity_temperature"]),
                    score_pooling=str(method_config.get("random_walk_score_pooling", "average")),
                )
                refined = propagated.clamp(0.0, 1.0)
                if bool(method_config.get("preserve_raw_foreground", False)):
                    refined = torch.maximum(refined, teacher_probability)
                if self.stage2_mode != "none" and case_name is not None:
                    self.stage2_memory.update(case_name, features, teacher_probability)
                del features, teacher_logits, teacher_probability, propagated
            self.model.train()
            with self._autocast():
                student_logits = self.model(teacher_image, self.embedding)
            student_features = None
        student_probability = torch.sigmoid(student_logits.float())
        # Once a case passes Eq. (11), the paper supervises the complete
        # affinity-refined pseudo-label rather than masking it a second time.
        weight = torch.ones_like(refined)
        if self.high_voxel_weighting_enabled:
            if voxel_confidence is None:
                raise ValueError("High voxel weighting is enabled but no aligned confidence patch was supplied")
            if tuple(voxel_confidence.shape) != tuple(image.shape):
                raise ValueError(
                    f"Voxel confidence/image shape mismatch: {tuple(voxel_confidence.shape)} vs {tuple(image.shape)}"
                )
            if tuple(voxel_confidence.shape) != tuple(refined.shape):
                raise ValueError(
                    f"Voxel confidence/target shape mismatch: {tuple(voxel_confidence.shape)} vs {tuple(refined.shape)}"
                )
            weighting = self.high_voxel_weighting
            weight = confidence_weight_map(
                voxel_confidence,
                function=str(weighting.get("function", "linear")),
                floor=float(weighting.get("floor", 0.3)),
                scale=float(weighting.get("scale", 0.7)),
                power=float(weighting.get("power", 1.0)),
            )
        if self.high_pseudo_cache is None:
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
        if self.high_relation_enabled and student_features is not None and case_name is not None:
            prototype_items = self.high_pseudo_cache.prototypes
            candidates = [(name, value) for name, value in prototype_items.items() if name != case_name]
            if candidates:
                mask = (refined >= threshold).float()
                if tuple(mask.shape[-3:]) != tuple(student_features.shape[-3:]):
                    mask = F.adaptive_max_pool3d(mask, student_features.shape[-3:])
                denominator = mask.sum(dim=(2, 3, 4)).clamp_min(1.0)
                student_pool = (student_features.float() * mask).sum(dim=(2, 3, 4)) / denominator
                student_pool = F.normalize(student_pool, dim=1, eps=1e-6)
                matrix = torch.stack([value.to(device=self.device, dtype=torch.float32) for _, value in candidates])
                similarities = student_pool @ matrix.T
                k = min(self.high_relation_top_k, len(candidates))
                values, indices = torch.topk(similarities, k=k, dim=1)
                weights = torch.softmax(values / self.high_relation_temperature, dim=1)
                reference = (matrix[indices] * weights.unsqueeze(-1)).sum(dim=1)
                reference = F.normalize(reference, dim=1, eps=1e-6).detach()
                relation_loss = (1.0 - F.cosine_similarity(student_pool, reference, dim=1)).mean()
                loss = loss + self.high_relation_weight * relation_loss
                patch_metrics["relation_loss"] = float(relation_loss.detach().item())
        return loss, patch_metrics

    def _high_confidence_patch(
        self,
        case: CaseRecord,
        slicer: tuple[slice, slice, slice],
        expected_volume_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Load cached confidence using the exact padded image-volume slicer."""
        if self.high_confidence_cache is None:
            raise RuntimeError("High voxel confidence cache is not configured")
        if self.high_confidence_memory_case != case.name:
            self.high_confidence_case_memory = self.high_confidence_cache.load_confidence(case)
            self.high_confidence_memory_case = case.name
        assert self.high_confidence_case_memory is not None
        data = self.high_confidence_case_memory
        confidence = data["joint_voxel_confidence"]
        geometry = data["geometry"]
        cached_shape = tuple(int(value) for value in confidence.shape[-3:])
        geometry_shape = tuple(int(value) for value in geometry.get("padded_shape", ()))
        if cached_shape != tuple(expected_volume_shape) or cached_shape != geometry_shape:
            raise ValueError(
                f"Confidence cache geometry mismatch for {case.name}: "
                f"array={cached_shape}, metadata={geometry_shape}, image={tuple(expected_volume_shape)}"
            )
        patch = np.ascontiguousarray(confidence[(slice(None), *slicer)])
        tensor = torch.from_numpy(patch).unsqueeze(0).to(self.device, dtype=torch.float32)
        if tuple(tensor.shape[-3:]) != self.patch_size:
            raise ValueError(f"Confidence patch has unexpected shape for {case.name}: {tuple(tensor.shape)}")
        return tensor

    def _low_confidence_loss(
        self,
        image: torch.Tensor,
        partner: torch.Tensor | None,
        case: CaseRecord | None = None,
        case_slicer: tuple[slice, slice, slice] | None = None,
    ) -> tuple[torch.Tensor, dict[str, int | bool]]:
        if bool(self.config["train"].get("low_raw_frozen_source", False)):
            return self._low_raw_frozen_source_loss(image)
        if partner is None:
            raise ValueError("The existing low-view consistency branch requires its partner image")
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
        language_diagnostics = None
        # The language-margin branch must use VoxTell's final mask response.
        # Reusing this forward's graph is important on 11 GB cards: a second
        # diagnostic forward while the consistency graph is alive OOMs during
        # backward.  The response is aligned to the original patch, so this
        # branch intentionally uses the unaugmented image as its student view.
        language_margin_enabled = self.consensus_mode in {
            "joint_language",
            "dynamic_hard_language",
        }
        if language_margin_enabled:
            with self._autocast():
                student_logits, language_diagnostics = self.model(
                    image, self.embedding, return_diagnostics=True
                )
            student_probability = torch.sigmoid(student_logits.float())
        else:
            with self._autocast():
                if self.stage2_mode != "none":
                    student_logits, student_features = self.model(
                        student_view.image, self.embedding, return_features=True
                    )
                else:
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
        low_relation_value = 0.0
        if self.low_relation_enabled:
            with self._autocast():
                _, relation_features = self.model(image, self.embedding, return_features=True)
            low_relation, low_relation_value = self._low_memory_relation_term(
                relation_features, prototype
            )
            if low_relation is not None:
                loss = loss + self.low_relation_weight * low_relation
            del relation_features
        consensus_extra: dict[str, float] = {}
        if self.consensus_enabled and case is not None and case_slicer is not None:
            consensus_patch = self._consensus_patch(case, case_slicer)
            field = self._consensus_probability_field()
            confidence_field = self._consensus_confidence_field()
            target = consensus_patch[field].to(self.device).unsqueeze(0)
            confidence = consensus_patch[confidence_field].to(self.device).unsqueeze(0).clamp(0.0, 1.0)
            threshold = float(self.config.get("consensus", {}).get("confidence_threshold", 0.0))
            if threshold > 0.0:
                confidence = confidence * (confidence >= threshold).float()
            consensus_loss = masked_binary_cross_entropy(
                student_probability,
                target.detach(),
                confidence.detach().clamp_min(1e-3),
            )
            weight = float(self.config.get("consensus", {}).get("lambda_consensus", 0.05))
            loss = loss + weight * consensus_loss
            consensus_extra["consensus_loss"] = float(consensus_loss.detach().item())
            if language_margin_enabled:
                language_loss, language_metrics = self._language_margin_loss(
                    image, consensus_patch, diagnostics=language_diagnostics
                )
                loss = loss + float(self.config.get("consensus", {}).get("lambda_language", 0.02)) * language_loss
                consensus_extra.update(language_metrics)
        stage2_extra: dict[str, float] = {}
        if self.stage2_mode != "none" and self.stage2_memory.entries:
            # Prior-Fixed is intentionally the old implementation: weighted
            # prototypes are concatenated before normalization and the legacy
            # foreground-only attraction is enabled only for this ablation.
            if self.stage2_mode in {"prior_fixed", "prior", "global", "local", "full"}:
                retrieved = self.stage2_memory.retrieve(student_features, self.stage2_top_k, self.stage2_temperature)
                supports = []
            else:
                # Correct path: each support independently produces A_k(v),
                # then case-level query/support weights fuse the priors.
                retrieved = None
                supports = self.stage2_memory.retrieve_supports(student_features, self.stage2_top_k, self.stage2_temperature)
            if retrieved is not None or supports:
                if retrieved is not None:
                    foreground, background, boundary = retrieved
                    prior = self.stage2_memory.prior(student_features.detach(), foreground, background, self.stage2_temperature)
                else:
                    foreground = background = boundary = None
                    prior = self.stage2_memory.weighted_prior(student_features.detach(), supports, self.stage2_temperature)
                prior = F.interpolate(prior, size=student_probability.shape[-3:], mode="trilinear", align_corners=False).detach()
                uncertain = (student_probability > self.stage2_uncertain_low) & (student_probability < self.stage2_uncertain_high)
                prior_loss = F.binary_cross_entropy(
                    student_probability.clamp(1e-5, 1.0 - 1e-5), prior, reduction="none"
                )
                prior_loss = (prior_loss * uncertain.float()).sum() / uncertain.sum().clamp_min(1)
                prior_loss = torch.nan_to_num(prior_loss)

                # Legacy attraction is kept only to reproduce the old Prior
                # result. It is disabled by default for all corrected modes.
                proto_loss = student_logits.sum() * 0.0
                if self.stage2_legacy_proto and retrieved is not None:
                    pooled_features = F.adaptive_avg_pool3d(torch.nan_to_num(student_features.float()), (8, 8, 8))
                    q_features = F.normalize(pooled_features, dim=1, eps=1e-6)
                    uncertain_feature = F.interpolate(uncertain.float(), size=q_features.shape[-3:], mode="nearest")[:, 0].bool()
                    q_vectors = q_features.permute(0, 2, 3, 4, 1)[uncertain_feature]
                    if q_vectors.shape[0] > 512:
                        q_vectors = q_vectors[:: max(1, q_vectors.shape[0] // 512)][:512]
                    fg_center = F.normalize(foreground.to(q_features.device).mean(0), dim=0, eps=1e-6)
                    if self.stage2_refine:
                        query_fg = student_probability.detach() >= 0.7
                        query_fg = F.interpolate(query_fg.float(), size=q_features.shape[-3:], mode="nearest")[:, 0].bool()
                        query_vectors_fg = q_features.permute(0, 2, 3, 4, 1)[query_fg]
                        if query_vectors_fg.numel() > 0:
                            fg_center = F.normalize(0.7 * fg_center + 0.3 * query_vectors_fg.mean(0), dim=0, eps=1e-6)
                    if q_vectors.numel() > 0:
                        proto_loss = (1.0 - F.cosine_similarity(q_vectors, fg_center[None], dim=1, eps=1e-6)).mean()
                    proto_loss = torch.nan_to_num(proto_loss)

                boundary_loss = student_logits.sum() * 0.0
                if self.stage2_mode == "clustered_boundary":
                    boundary_prior = self.stage2_memory.weighted_prior(
                        student_features.detach(), supports, self.stage2_temperature, field="boundary"
                    )
                    boundary_target = F.interpolate(
                        boundary_prior, size=student_probability.shape[-3:], mode="trilinear", align_corners=False
                    ).detach()
                    boundary_map = F.max_pool3d((student_probability > 0.5).float(), 3, 1, 1) - (-F.max_pool3d(-(student_probability > 0.5).float(), 3, 1, 1))
                    boundary_mask = (boundary_map.abs() > 0).float()
                    boundary_loss = F.binary_cross_entropy(
                        student_probability.clamp(1e-5, 1.0 - 1e-5), boundary_target, reduction="none"
                    )
                    boundary_loss = torch.nan_to_num((boundary_loss * boundary_mask).sum() / boundary_mask.sum().clamp_min(1))

                if self.stage2_mode in {"prior_fixed", "prior"}:
                    loss = loss + self.stage2_proto_weight * proto_loss + self.stage2_prior_weight * prior_loss
                elif self.stage2_mode in {"weighted_prior", "clustered_prior"}:
                    loss = loss + self.stage2_prior_weight * prior_loss
                elif self.stage2_mode == "clustered_boundary":
                    loss = loss + self.stage2_prior_weight * prior_loss + self.stage2_boundary_weight * boundary_loss
                elif self.stage2_mode in {"global", "local", "full"}:
                    # Backwards-compatible aliases for previously generated
                    # runs; new experiments use the explicit names above.
                    loss = loss + self.stage2_proto_weight * proto_loss + self.stage2_prior_weight * prior_loss
                stage2_extra["stage2_prior_mean"] = float(prior.detach().mean().item())
        threshold = float(self.config["method"]["probability_threshold"])
        prototype_voxels = int((prototype >= threshold).sum().item())
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss, {
            "raw_foreground_voxels": prototype_voxels,
            "refined_foreground_voxels": prototype_voxels,
            "voxels": prototype.numel(),
            "effective": True,
            "relation_loss": low_relation_value,
            **stage2_extra,
            **consensus_extra,
        }

    def _low_raw_frozen_source_loss(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, int | bool]]:
        """Raw-image low baseline: frozen source Bernoulli target, no view generation."""
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad(), lora_disabled(self.model):
            with self._autocast():
                teacher_logits = self.model(image, self.embedding)
            teacher_probability = torch.sigmoid(teacher_logits.float())
            del teacher_logits
        self.model.train(was_training)
        if self.low_relation_enabled:
            if self.low_relation_checkpointing:
                def student_forward(input_image: torch.Tensor):
                    with self._autocast():
                        return self.model(input_image, self.embedding, return_features=True)
                student_logits, student_features = activation_checkpoint(
                    student_forward, image, use_reentrant=False
                )
            else:
                with self._autocast():
                    student_logits, student_features = self.model(
                        image, self.embedding, return_features=True
                    )
        else:
            with self._autocast():
                student_logits = self.model(image, self.embedding)
            student_features = None
        student_probability = torch.sigmoid(student_logits.float())
        del student_logits
        train_config = self.config["train"]
        consistency = bernoulli_kl_divergence(
            teacher_probability,
            student_probability,
            balanced=bool(train_config.get("balance_low_consistency", True)),
        )
        entropy_weight = float(train_config["low_entropy_weight"])
        loss = float(train_config["low_consistency_weight"]) * consistency
        if entropy_weight != 0.0:
            loss = loss + entropy_weight * binary_entropy(student_probability)
        del student_probability, consistency
        relation_value = 0.0
        if student_features is not None:
            relation, relation_value = self._low_memory_relation_term(
                student_features, teacher_probability
            )
            if relation is not None:
                loss = loss + self.low_relation_weight * relation
            del student_features
        threshold = float(self.config["method"]["probability_threshold"])
        foreground_voxels = int((teacher_probability >= threshold).sum().item())
        voxel_count = int(teacher_probability.numel())
        del teacher_probability
        metrics = {
            "raw_foreground_voxels": foreground_voxels,
            "refined_foreground_voxels": foreground_voxels,
            "voxels": voxel_count,
            "effective": True,
            "relation_loss": relation_value,
        }
        return torch.nan_to_num(loss), metrics

    def _low_memory_relation_term(
        self, student_features: torch.Tensor, teacher_probability: torch.Tensor
    ) -> tuple[torch.Tensor | None, float]:
        prototypes = self.high_pseudo_cache.prototypes
        if not prototypes:
            return None, 0.0
        prototype_matrix = torch.stack(
            [value.to(device=self.device, dtype=torch.float32) for value in prototypes.values()]
        )
        feature_map = F.normalize(student_features.float(), dim=1, eps=1e-6)
        feature_vectors = feature_map.permute(0, 2, 3, 4, 1)
        memory_similarity = torch.einsum("bdhwc,pc->bpdhw", feature_vectors, prototype_matrix)
        best_similarity = memory_similarity.max(dim=1).values
        foreground = (teacher_probability >= float(self.config["method"]["probability_threshold"])).float()
        if tuple(foreground.shape[-3:]) != tuple(best_similarity.shape[-3:]):
            foreground = F.adaptive_max_pool3d(foreground, best_similarity.shape[-3:])
        foreground = foreground[:, 0].bool()
        if not bool(foreground.any()):
            return None, 0.0
        relation = (1.0 - best_similarity[foreground]).mean()
        return relation, float(relation.detach().item())

    def _compute_high_reference_gradients(self, iteration: int) -> dict[int, torch.Tensor | None]:
        if not self.high_reference_specs:
            raise RuntimeError("High reference patches were not initialized")
        case, patch_index = self.high_reference_specs[(iteration - 1) % len(self.high_reference_specs)]
        volume, slicers = self.store.sliding_window(case, self.tile_step_size)
        if patch_index >= len(slicers):
            raise RuntimeError(f"Cached High patch index is invalid for {case.name}")
        image = self._patch(volume, slicers[patch_index])
        loss, _ = self._high_confidence_loss(image, case.name, patch_index)
        gradients = torch.autograd.grad(
            loss,
            self.runtime.trainable_parameters,
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )
        reference = {
            id(parameter): (None if gradient is None else gradient.detach().float())
            for parameter, gradient in zip(self.runtime.trainable_parameters, gradients)
        }
        del volume, image, loss, gradients
        return reference

    def _apply_optimizer_routing(
        self,
        high_reference_gradients: dict[int, torch.Tensor | None] | None,
        iteration: int,
    ) -> str:
        if high_reference_gradients is None:
            raise RuntimeError("Missing independent High reference gradients")
        group_names = sorted(self.parameter_groups)
        group_count = len(group_names)
        keep_count = min(group_count, max(1, int(np.ceil(group_count * self.optimizer_routing_fraction))))
        scores: dict[str, float] = {}
        score_is_finite: dict[str, bool] = {}
        for name in group_names:
            low_parts = []
            high_parts = []
            for parameter in self.parameter_groups[name]:
                if parameter.grad is None:
                    low_parts.append(torch.zeros(parameter.numel(), device=self.device))
                else:
                    low_parts.append(parameter.grad.detach().float().reshape(-1))
                reference = high_reference_gradients.get(id(parameter))
                if reference is None:
                    high_parts.append(torch.zeros(parameter.numel(), device=self.device))
                else:
                    high_parts.append(reference.reshape(-1))
            low_vector = torch.cat(low_parts)
            high_vector = torch.cat(high_parts)
            if self.optimizer_routing_mode == "norm":
                score = float(torch.linalg.vector_norm(low_vector).item())
            elif self.optimizer_routing_mode == "cosine":
                low_norm = torch.linalg.vector_norm(low_vector)
                high_norm = torch.linalg.vector_norm(high_vector)
                if float(low_norm.item()) <= 1e-12 or float(high_norm.item()) <= 1e-12:
                    score = -1.0
                else:
                    score = float(F.cosine_similarity(low_vector, high_vector, dim=0).item())
            else:
                score = 0.0
            score_is_finite[name] = bool(np.isfinite(score))
            # Keep top-k ordering deterministic when mixed-precision gradients
            # contain NaN/Inf. Finite groups rank first; if all groups are
            # invalid, the sorted group names provide a stable fallback.
            scores[name] = score if score_is_finite[name] else float("-inf")

        if self.optimizer_routing_mode == "fixed":
            selected = set(group_names[:keep_count])
        elif self.optimizer_routing_mode == "random":
            order = torch.randperm(group_count, generator=self.route_generator).tolist()
            selected = {group_names[index] for index in order[:keep_count]}
        else:
            selected = set(sorted(group_names, key=lambda name: (-scores[name], name))[:keep_count])

        for name in group_names:
            chosen = name in selected
            for parameter in self.parameter_groups[name]:
                if not chosen:
                    # None prevents AdamW momentum and weight decay from changing a gated group.
                    parameter.grad = None
            self.parameter_route_rows.append({
                "iteration": int(iteration),
                "mode": self.optimizer_routing_mode,
                "group_fraction": self.optimizer_routing_fraction,
                "group": name,
                "score": scores[name],
                "score_finite": score_is_finite[name],
                "selected": chosen,
            })
        return ",".join(sorted(selected))

    def _write_parameter_routing(self) -> None:
        if not self.optimizer_routing_enabled:
            return
        path = self.output_dir / "parameter_routing.csv"
        fields = ("iteration", "mode", "group_fraction", "group", "score", "score_finite", "selected")
        with path.open("w", encoding="utf-8", newline="") as handle:
            import csv
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.parameter_route_rows)

    def _consensus_probability_field(self) -> str:
        if self.consensus_mode == "image":
            return "image_consensus_probability"
        if self.consensus_mode == "prompt":
            return "prompt_consensus_probability"
        return "joint_probability"

    def _consensus_confidence_field(self) -> str:
        if self.consensus_mode == "image":
            return "image_voxel_consistency"
        if self.consensus_mode == "prompt":
            return "prompt_voxel_consistency"
        return "joint_voxel_confidence"

    def _consensus_patch(self, case: CaseRecord, slicer: tuple[slice, slice, slice]) -> dict[str, torch.Tensor]:
        if self.consensus_cache is None:
            raise RuntimeError("consensus cache is not initialized")
        if self.consensus_memory_case != case.name:
            self.consensus_case_memory = self.consensus_cache.load_case(case)
            self.consensus_memory_case = case.name
        result = {}
        for field in (
            "image_consensus_probability",
            "prompt_consensus_probability",
            "image_voxel_consistency",
            "prompt_voxel_consistency",
            "joint_voxel_confidence",
            "joint_probability",
        ):
            array = self.consensus_case_memory[field]
            result[field] = torch.from_numpy(np.ascontiguousarray(array[(slice(None), *slicer)]))
        return result

    def _language_margin_loss(
        self,
        image: torch.Tensor,
        consensus_patch: dict[str, torch.Tensor],
        diagnostics: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Apply the margin to VoxTell's own final mask response, not raw text/D5."""
        if diagnostics is None:
            with self._autocast():
                _, diagnostics = self.model(image, self.embedding, return_diagnostics=True)
        response = diagnostics.get("mask_response")
        if response is None:
            raise RuntimeError("VoxTell runtime did not return final mask response diagnostics")
        response = response[:, 0].float()
        probability = consensus_patch["joint_probability"].to(self.device).unsqueeze(0)[:, 0]
        confidence = consensus_patch["joint_voxel_confidence"].to(self.device).unsqueeze(0)[:, 0]
        cfg = self.config.get("consensus", {})
        fg = (probability >= float(cfg.get("tau_fg", 0.7))) & (confidence >= float(cfg.get("tau_conf", 0.5)))
        bg = (probability <= float(cfg.get("tau_bg", 0.3))) & (confidence >= float(cfg.get("tau_conf", 0.5)))
        min_voxels = int(cfg.get("language_min_voxels", 8))
        if int(fg.sum()) < min_voxels or int(bg.sum()) < min_voxels:
            return response.sum() * 0.0, {"language_fg_voxels": float(fg.sum()), "language_bg_voxels": float(bg.sum()), "language_loss": 0.0}
        r_fg = response[fg].mean()
        r_bg = response[bg].mean()
        margin = float(cfg.get("margin", 0.1))
        loss = F.relu(response.new_tensor(margin) - r_fg + r_bg)
        return loss, {"language_fg_voxels": float(fg.sum()), "language_bg_voxels": float(bg.sum()), "language_loss": float(loss.detach().item())}

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
            save_predictions=bool(self.config["train"].get("save_validation_predictions", False)),
        )
        assisted = None
        if self.sat_proposal_dir:
            assisted = evaluate_sat_assisted(self.config, self.runtime, self.validation_cases, self.validation_gt_path, self.sat_proposal_dir, self.sat_gate_threshold, validation_dir / "sat_assisted", self.logger, save_predictions=True, gate_mode=self.sat_gate_mode)
        self.model.train(was_training)
        self.last_validation_iteration = iteration
        mean_dice = float(summary["mean_dice"])
        if mean_dice > self.best_dice:
            self.patience_counter = 0
            self._save_best(iteration, mean_dice)
        else:
            self.patience_counter += 1
        if assisted is not None and assisted["assisted_mean_dice"] > self.best_sat_dice:
            self.best_sat_dice = float(assisted["assisted_mean_dice"])
            self._save(self.output_dir / "checkpoints" / "best_sat_assisted.pt", iteration)
        self._save_latest(iteration)
        self.logger.info(
            "validation_iteration=%d mean_dice=%.6f mean_iou=%.6f",
            iteration,
            summary["mean_dice"],
            summary["mean_iou"],
        )
        if assisted is not None:
            self.logger.info("validation_iteration=%d SAT-assisted mean_dice=%.6f mean_iou=%.6f gate_pass=%d", iteration, assisted["assisted_mean_dice"], assisted["assisted_mean_iou"], assisted["sat_gate_pass"])
        patience = int(self.config["train"].get("early_stopping_patience", 10))
        should_stop = patience > 0 and self.patience_counter >= patience
        self._save_early_stopping()
        return should_stop

    def _save_early_stopping(self) -> None:
        save_json(
            {
                "best_dice": None if self.best_dice == float("-inf") else self.best_dice,
                "best_iteration": self.best_iteration,
                "patience_counter": self.patience_counter,
                "early_stopping_patience": int(self.config["train"].get("early_stopping_patience", 10)),
                "stop_iteration": self.stop_iteration,
                "stop_reason": self.stop_reason,
            },
            self.output_dir / "early_stopping.json",
        )

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
            trainer_state={
                "patience_counter": int(self.patience_counter),
                "stop_iteration": self.stop_iteration,
                "stop_reason": self.stop_reason,
                "refresh_cycle": int(self.refresh_cycle),
                "seen_cases_since_refresh": sorted(self.seen_cases_since_refresh),
                "refresh_history": self.refresh_history,
                "consensus_mode": self.consensus_mode,
            },
            scaler_state=self.scaler.state_dict() if self.amp_enabled else None,
        )
        self._save_early_stopping()
        self._write_refresh_history()
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
        self._save(checkpoint_dir / "best_voxtell_only.pt", iteration)
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
        checkpoint = load_checkpoint(path, self.model, self.optimizer)
        configured_lr = float(self.config["train"]["learning_rate"])
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = configured_lr
        self.start_iteration = int(checkpoint.get("iteration", 0))
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
        trainer_state = checkpoint.get("trainer_state", {})
        if self.amp_enabled and checkpoint.get("scaler_state"):
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.patience_counter = int(trainer_state.get("patience_counter", 0))
        self.stop_iteration = trainer_state.get("stop_iteration")
        self.stop_reason = trainer_state.get("stop_reason")
        self.refresh_cycle = int(trainer_state.get("refresh_cycle", 0))
        self.seen_cases_since_refresh = set(str(v) for v in trainer_state.get("seen_cases_since_refresh", []))
        self.refresh_history = list(trainer_state.get("refresh_history", []))
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
            trainer_state = rng_state.get("trainer_generator")
            if trainer_state is not None:
                self.generator.set_state(trainer_state)
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
