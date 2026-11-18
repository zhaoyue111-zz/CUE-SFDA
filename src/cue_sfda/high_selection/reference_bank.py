from __future__ import annotations

import copy

import numpy as np
import torch


class ReferenceBank:
    """CPU-resident case-indexed snapshot evidence for High update selection."""

    def __init__(self) -> None:
        self.gradients: dict[str, torch.Tensor] = {}
        self.features: dict[str, np.ndarray] = {}
        self.semantic_evidence: dict[str, dict] = {}
        self.semantic_cache_error: str | None = None
        self.reference_case_ids: list[str] = []
        self.generated_iteration: int | None = None
        self.model_fingerprint: str | None = None
        self.cache_version = 0

    def replace(
        self,
        *,
        gradients: dict[str, torch.Tensor],
        features: dict[str, np.ndarray],
        reference_case_ids: list[str],
        semantic_evidence: dict[str, dict] | None,
        semantic_cache_error: str | None = None,
        generated_iteration: int,
        model_fingerprint: str,
    ) -> None:
        self.gradients = {key: value.detach().cpu().clone() for key, value in gradients.items()}
        self.features = {
            key: np.asarray(value, dtype=np.float32).copy()
            for key, value in features.items()
        }
        self.reference_case_ids = sorted(str(value) for value in reference_case_ids)
        self.semantic_evidence = {}
        for key, value in (semantic_evidence or {}).items():
            self.semantic_evidence[key] = {
                **{field: item for field, item in value.items() if field != "basis"},
                "basis": torch.as_tensor(value["basis"], dtype=torch.float32).cpu().clone(),
            }
        self.semantic_cache_error = semantic_cache_error
        self.generated_iteration = int(generated_iteration)
        self.model_fingerprint = str(model_fingerprint)
        self.cache_version += 1

    def semantic_for(self, candidate_case: str) -> dict:
        key = candidate_case if candidate_case in self.features else "__external_candidate__"
        if key not in self.semantic_evidence:
            raise ValueError(f"semantic reference evidence unavailable for {candidate_case}")
        return self.semantic_evidence[key]

    def gradient_references(self, excluded_case: str, device: torch.device) -> list[torch.Tensor]:
        return [value.to(device) for key, value in self.gradients.items() if key != excluded_case]

    def feature_references(self, excluded_case: str) -> dict[str, np.ndarray]:
        return {key: value for key, value in self.features.items() if key != excluded_case}

    def state_dict(self) -> dict:
        return {
            "gradients": {key: value.detach().cpu().clone() for key, value in self.gradients.items()},
            "features": {key: value.copy() for key, value in self.features.items()},
            "semantic_evidence": copy.deepcopy(self.semantic_evidence),
            "semantic_cache_error": self.semantic_cache_error,
            "reference_case_ids": list(self.reference_case_ids),
            "generated_iteration": self.generated_iteration,
            "model_fingerprint": self.model_fingerprint,
            "cache_version": self.cache_version,
        }

    def load_state_dict(self, state: dict) -> None:
        required = {"gradients", "features", "semantic_evidence", "semantic_cache_error", "reference_case_ids", "generated_iteration", "model_fingerprint", "cache_version"}
        missing = required.difference(state)
        if missing:
            raise ValueError(f"reference bank state missing fields: {sorted(missing)}")
        self.gradients = {key: torch.as_tensor(value).detach().cpu().clone() for key, value in state["gradients"].items()}
        self.features = {key: np.asarray(value, dtype=np.float32).copy() for key, value in state["features"].items()}
        self.reference_case_ids = sorted(str(value) for value in state["reference_case_ids"])
        self.semantic_evidence = {}
        self.semantic_cache_error = state["semantic_cache_error"]
        for key, value in state["semantic_evidence"].items():
            basis = torch.as_tensor(value["basis"], dtype=torch.float32).cpu().clone()
            if basis.ndim != 2 or not bool(torch.isfinite(basis).all()):
                raise ValueError(f"invalid semantic cache basis for {key}")
            for field in ("threshold", "spectral_energy"):
                if not np.isfinite(float(value[field])):
                    raise ValueError(f"invalid semantic cache {field} for {key}")
            if int(value["rank"]) != basis.shape[1] or int(value["reference_case_count"]) < 2:
                raise ValueError(f"invalid semantic cache rank/count for {key}")
            self.semantic_evidence[key] = {**value, "basis": basis}
        self.generated_iteration = state["generated_iteration"]
        self.model_fingerprint = state["model_fingerprint"]
        self.cache_version = int(state["cache_version"])
        if self.cache_version < 0:
            raise ValueError("reference bank cache version cannot be negative")
        if self.generated_iteration is None and (self.gradients or self.features or self.semantic_evidence):
            raise ValueError("reference bank has evidence without a generation iteration")
        if self.cache_version > 0 and (self.generated_iteration is None or not self.model_fingerprint):
            raise ValueError("versioned reference bank is missing snapshot metadata")
        if not all(bool(torch.isfinite(value).all()) for value in self.gradients.values()):
            raise ValueError("reference bank contains non-finite gradients")
        if not all(np.isfinite(value).all() for value in self.features.values()):
            raise ValueError("reference bank contains non-finite features")
        if not set(self.gradients).issubset(self.reference_case_ids) or not set(self.features).issubset(self.reference_case_ids):
            raise ValueError("reference evidence contains an unlisted case ID")
        if any(key != "__external_candidate__" and key not in self.reference_case_ids for key in self.semantic_evidence):
            raise ValueError("semantic cache contains an unlisted candidate exclusion ID")
        if self.features and self.semantic_cache_error is None:
            required_semantics = set(self.features) | {"__external_candidate__"}
            if not required_semantics.issubset(self.semantic_evidence):
                raise ValueError("semantic cache is incomplete and has no recorded build error")
