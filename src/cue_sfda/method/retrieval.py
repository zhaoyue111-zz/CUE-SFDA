from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x.float()), dim=-1, eps=1e-6)


@dataclass
class SupportEntry:
    global_feature: torch.Tensor
    foreground: torch.Tensor
    background: torch.Tensor
    boundary: torch.Tensor


class StableSupportMemory:
    """CPU support memory written only by Stable/high-confidence cases."""

    def __init__(self, max_local: int = 4, momentum: float = 0.9, clustered: bool = True) -> None:
        self.max_local = max(1, int(max_local))
        self.momentum = float(momentum)
        self.clustered = bool(clustered)
        self.entries: dict[str, SupportEntry] = {}

    @staticmethod
    def _cluster(values: torch.Tensor, count: int) -> torch.Tensor:
        """Deterministic, bounded cosine farthest-point clustering."""
        values = _normalize(values)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("prototype clustering requires non-empty [N,C] values")
        if values.shape[0] > 512:
            idx = torch.linspace(0, values.shape[0] - 1, 512, device=values.device).long()
            values = values[idx]
        k = min(max(1, int(count)), values.shape[0])
        centers = [values.mean(dim=0)]
        for _ in range(1, k):
            center_matrix = _normalize(torch.stack(centers, dim=0))
            distance = 1.0 - values @ center_matrix.T
            centers.append(values[distance.min(dim=1).values.argmax()])
        centers_tensor = _normalize(torch.stack(centers, dim=0))
        for _ in range(3):
            assignments = (values @ centers_tensor.T).argmax(dim=1)
            updated = []
            for index in range(k):
                assigned = values[assignments == index]
                updated.append(assigned.mean(dim=0) if assigned.numel() else centers_tensor[index])
            centers_tensor = _normalize(torch.stack(updated, dim=0))
        return centers_tensor.cpu()

    @classmethod
    def _make_prototypes(
        cls, features: torch.Tensor, mask: torch.Tensor, count: int, local: bool = True, clustered: bool = True
    ) -> torch.Tensor:
        values = features.permute(1, 2, 3, 0)[mask]
        if values.numel() == 0:
            return features.new_empty((0, features.shape[0]))
        if local:
            if not clustered:
                _, d, h, w = features.shape
                parts = [mask]
                for di in range(2):
                    for hi in range(2):
                        for wi in range(2):
                            part = torch.zeros_like(mask)
                            part[di * d // 2:(di + 1) * d // 2,
                                 hi * h // 2:(hi + 1) * h // 2,
                                 wi * w // 2:(wi + 1) * w // 2] = mask[di * d // 2:(di + 1) * d // 2,
                                                                          hi * h // 2:(hi + 1) * h // 2,
                                                                          wi * w // 2:(wi + 1) * w // 2]
                            parts.append(part)
                prototypes = []
                for part in parts:
                    part_values = features.permute(1, 2, 3, 0)[part]
                    if part_values.numel():
                        prototypes.append(_normalize(part_values.mean(dim=0, keepdim=True)).squeeze(0))
                if prototypes:
                    return torch.stack(prototypes[:max(1, int(count))], dim=0).cpu()
            return cls._cluster(values, count)
        return _normalize(values.mean(dim=0, keepdim=True)).cpu()

    def update(self, case_id: str, feature_map: torch.Tensor, probability: torch.Tensor) -> None:
        feature_map = torch.nan_to_num(feature_map.detach().float())[0].cpu()
        probability = torch.nan_to_num(probability.detach().float()).cpu()
        if probability.ndim == 5:
            probability = probability[0, 0]
        elif probability.ndim == 4:
            probability = probability[0]
        coarse = F.interpolate(
            probability[None, None], size=feature_map.shape[-3:], mode="trilinear", align_corners=False
        )[0, 0]
        fg = coarse >= 0.7
        bg = coarse <= 0.2
        hard = coarse >= 0.5
        dilated = F.max_pool3d(hard[None, None].float(), 3, 1, 1)[0, 0] > 0
        eroded = -F.max_pool3d((-hard[None, None].float()), 3, 1, 1)[0, 0] > 0.5
        boundary = dilated ^ eroded
        global_feature = _normalize(feature_map.flatten(1).mean(dim=1, keepdim=True).T).squeeze(0).cpu()
        foreground = self._make_prototypes(feature_map, fg, self.max_local, local=True, clustered=self.clustered)
        background = self._make_prototypes(feature_map, bg, self.max_local, local=True, clustered=self.clustered)
        fallback = global_feature[None]
        if foreground.shape[0] == 0:
            foreground = fallback.clone()
        if background.shape[0] == 0:
            background = fallback.clone()
        boundary_proto = self._make_prototypes(feature_map, boundary, 1, local=False)
        if boundary_proto.shape[0] == 0:
            boundary_proto = fallback.clone()
        entry = SupportEntry(global_feature, foreground, background, boundary_proto)
        old = self.entries.get(case_id)
        if old is None:
            self.entries[case_id] = entry
            return

        def ema(old_value: torch.Tensor, new_value: torch.Tensor) -> torch.Tensor:
            n = min(old_value.shape[0], new_value.shape[0])
            return _normalize(self.momentum * old_value[:n] + (1.0 - self.momentum) * new_value[:n]).cpu()

        self.entries[case_id] = SupportEntry(
            global_feature=_normalize(self.momentum * old.global_feature + (1.0 - self.momentum) * entry.global_feature).cpu(),
            foreground=ema(old.foreground, entry.foreground),
            background=ema(old.background, entry.background),
            boundary=ema(old.boundary, entry.boundary),
        )

    def _ranked_supports(self, query_feature: torch.Tensor, top_k: int, temperature: float) -> tuple[list[str], torch.Tensor] | None:
        if not self.entries:
            return None
        query_global = _normalize(query_feature.detach().float().flatten(2).mean(dim=2)).squeeze(0)
        names = list(self.entries)
        globals_ = torch.stack([self.entries[name].global_feature for name in names], dim=0).to(query_feature.device)
        scores = globals_ @ query_global.to(globals_.device)
        k = min(max(1, int(top_k)), len(names))
        values, indices = torch.topk(scores, k=k)
        weights = torch.softmax(values / max(float(temperature), 1e-4), dim=0)
        return [names[index] for index in indices.tolist()], weights

    def retrieve(self, query_feature: torch.Tensor, top_k: int, temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Legacy weighted-prototype retrieval used only by Prior-Fixed."""
        ranked = self._ranked_supports(query_feature, top_k, temperature)
        if ranked is None:
            return None
        names, weights = ranked
        return tuple(torch.cat([
            getattr(self.entries[name], field).to(query_feature.device) * weights[i]
            for i, name in enumerate(names)
        ], dim=0) for field in ("foreground", "background", "boundary"))  # type: ignore[return-value]

    def retrieve_supports(self, query_feature: torch.Tensor, top_k: int, temperature: float) -> list[tuple[float, SupportEntry]]:
        """Return independent supports and query-support softmax weights."""
        ranked = self._ranked_supports(query_feature, top_k, temperature)
        if ranked is None:
            return []
        names, weights = ranked
        return [(float(weights[i].detach().cpu()), self.entries[name]) for i, name in enumerate(names)]

    @staticmethod
    def prior(feature_map: torch.Tensor, foreground: torch.Tensor, background: torch.Tensor, temperature: float) -> torch.Tensor:
        features = _normalize(feature_map.permute(0, 2, 3, 4, 1))
        fg = _normalize(foreground.to(feature_map.device)).T
        bg = _normalize(background.to(feature_map.device)).T
        fg_score = torch.einsum("bdhwc,cp->bdhwp", features, fg).max(dim=-1).values
        bg_score = torch.einsum("bdhwc,cp->bdhwp", features, bg).max(dim=-1).values
        return torch.nan_to_num(torch.sigmoid((fg_score - bg_score) / max(float(temperature), 1e-4))).unsqueeze(1)

    @classmethod
    def weighted_prior(cls, feature_map: torch.Tensor, supports: list[tuple[float, SupportEntry]], temperature: float, field: str = "foreground") -> torch.Tensor:
        """Compute each A_k first, then fuse A(v)=sum_k w_k A_k(v)."""
        if not supports:
            return feature_map.new_zeros((feature_map.shape[0], 1, *feature_map.shape[-3:]))
        priors = []
        for weight, entry in supports:
            if field == "boundary":
                features = _normalize(feature_map.permute(0, 2, 3, 4, 1))
                center = _normalize(entry.boundary.to(feature_map.device).mean(dim=0, keepdim=True)).squeeze(0)
                score = (features * center[None, None, None, None, :]).sum(dim=-1)
                prior = torch.sigmoid(score / max(float(temperature), 1e-4)).unsqueeze(1)
            else:
                prior = cls.prior(feature_map, entry.foreground, entry.background, temperature)
            priors.append((weight, prior))
        total_weight = sum(weight for weight, _ in priors)
        return torch.nan_to_num(sum(weight * prior for weight, prior in priors) / max(total_weight, 1e-6))
