from __future__ import annotations

from contextlib import nullcontext
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cue_sfda.data import CaseRecord, TargetVolumeStore
from cue_sfda.engine.checkpoint import load_checkpoint, save_checkpoint
from cue_sfda.evaluation import evaluate
from cue_sfda.method.augmentations import StrongViewGenerator
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
from cue_sfda.modeling.voxtell import VoxTellRuntime
from cue_sfda.modeling.lora import lora_disabled
from cue_sfda.utils import save_json
from cue_sfda.sat_rescue import evaluate_sat_assisted
from cue_sfda.method.retrieval import StableSupportMemory


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
        self.calibration_foreground_counts: dict[str, int] = {}
        self.skipped_case_names: set[str] = set()
        self.routing_metadata: dict[str, int | float | str] = {}
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
            metrics = self._train_step(case, partner)
            if iteration == 1 or iteration % log_every == 0:
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
                self._validate(iteration)
        if self.validation_cases and self.last_validation_iteration != total_iterations:
            self._validate(total_iterations)
        latest_path = self._save_latest(total_iterations)
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
        voxel_sum = 0
        effective_patch_count = 0
        skipped_patch_count = 0
        patch_count = len(case_slicers)
        for patch_index, case_slicer in enumerate(case_slicers):
            partner_slicer = partner_slicers[patch_index % len(partner_slicers)]
            image = self._patch(case_volume, case_slicer)
            partner_image = self._patch(partner_volume, partner_slicer)
            if high_confidence:
                loss, patch_metrics = self._high_confidence_loss(image, case.name)
                branch = "high"
            elif case.name in self.sat_rescue_cases:
                loss, patch_metrics = self._sat_rescue_loss(image, case, case_slicer, case_volume)
                branch = "sat_rescue"
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
            if self.stage2_mode != "none" and case_name is not None:
                self.stage2_memory.update(case_name, features, teacher_probability)
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
        self, image: torch.Tensor, partner: torch.Tensor
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
            **stage2_extra,
        }

    def _patch(
        self,
        volume: torch.Tensor,
        slicer: tuple[slice, slice, slice],
    ) -> torch.Tensor:
        patch = self.store.extract_patch(volume, slicer)
        return patch.unsqueeze(0).to(self.device, non_blocking=True)

    def _validate(self, iteration: int) -> None:
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
            save_predictions=True,
        )
        assisted = None
        if self.sat_proposal_dir:
            assisted = evaluate_sat_assisted(self.config, self.runtime, self.validation_cases, self.validation_gt_path, self.sat_proposal_dir, self.sat_gate_threshold, validation_dir / "sat_assisted", self.logger, save_predictions=True, gate_mode=self.sat_gate_mode)
        self.model.train(was_training)
        self.last_validation_iteration = iteration
        mean_dice = float(summary["mean_dice"])
        if mean_dice > self.best_dice:
            self._save_best(iteration, mean_dice)
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
        )
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
