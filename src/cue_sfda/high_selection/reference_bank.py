from __future__ import annotations

import numpy as np
import torch


class ReferenceBank:
    """CPU-resident case-indexed snapshot evidence for High update selection."""

    def __init__(self) -> None:
        self.gradients: dict[str, torch.Tensor] = {}
        self.features: dict[str, np.ndarray] = {}
        self.generated_iteration: int | None = None
        self.model_fingerprint: str | None = None

    def replace(
        self,
        *,
        gradients: dict[str, torch.Tensor],
        features: dict[str, np.ndarray],
        generated_iteration: int,
        model_fingerprint: str,
    ) -> None:
        self.gradients = {key: value.detach().cpu() for key, value in gradients.items()}
        self.features = {key: np.asarray(value, dtype=np.float32) for key, value in features.items()}
        self.generated_iteration = int(generated_iteration)
        self.model_fingerprint = str(model_fingerprint)

    def gradient_references(self, excluded_case: str, device: torch.device) -> list[torch.Tensor]:
        return [value.to(device) for key, value in self.gradients.items() if key != excluded_case]

    def feature_references(self, excluded_case: str) -> dict[str, np.ndarray]:
        return {key: value for key, value in self.features.items() if key != excluded_case}
