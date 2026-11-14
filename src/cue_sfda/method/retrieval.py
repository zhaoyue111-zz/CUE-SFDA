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
    """Small CPU memory populated exclusively by stable/high-quality cases."""

    def __init__(self, max_local: int = 4, momentum: float = 0.9) -> None:
        self.max_local = int(max_local)
        self.momentum = float(momentum)
        self.entries: dict[str, SupportEntry] = {}

    @staticmethod
    def _prototype(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
        values = features.permute(1, 2, 3, 0)[mask]
        if values.numel() == 0:
            return None
        return _normalize(values.mean(dim=0, keepdim=True)).squeeze(0)

    def _make_prototypes(
        self, features: torch.Tensor, mask: torch.Tensor, local: bool = True
    ) -> torch.Tensor:
        # Features are [C,D,H,W], masks are [D,H,W]. Spatial bins provide
        # deterministic local prototypes without an extra clustering network.
        _, d, h, w = features.shape
        masks = [mask]
        if local:
            for di in range(2):
                for hi in range(2):
                    for wi in range(2):
                        d0, d1 = di * d // 2, (di + 1) * d // 2
                        h0, h1 = hi * h // 2, (hi + 1) * h // 2
                        w0, w1 = wi * w // 2, (wi + 1) * w // 2
                        part = torch.zeros_like(mask)
                        part[d0:d1, h0:h1, w0:w1] = mask[d0:d1, h0:h1, w0:w1]
                        masks.append(part)
        result = []
        for part in masks:
            prototype = self._prototype(features, part)
            if prototype is not None:
                result.append(prototype)
        if not result:
            result.append(_normalize(features.flatten(1).mean(dim=1, keepdim=True).T).squeeze(0))
        return torch.stack(result[: self.max_local], dim=0).cpu()

    def update(
        self,
        case_id: str,
        feature_map: torch.Tensor,
        probability: torch.Tensor,
    ) -> None:
        # Only detached CPU tensors are retained. This makes memory updates
        # independent of the autograd graph and keeps GPU memory bounded.
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
        dilated = F.max_pool3d(hard[None, None].float(), kernel_size=3, stride=1, padding=1)[0, 0] > 0
        eroded = -F.max_pool3d((-hard[None, None].float()), kernel_size=3, stride=1, padding=1)[0, 0] > 0.5
        boundary = dilated ^ eroded
        entry = SupportEntry(
            global_feature=_normalize(feature_map.flatten(1).mean(dim=1, keepdim=True).T).squeeze(0).cpu(),
            foreground=self._make_prototypes(feature_map, fg, local=True),
            background=self._make_prototypes(feature_map, bg, local=True),
            boundary=self._make_prototypes(feature_map, boundary, local=False),
        )
        old = self.entries.get(case_id)
        if old is None:
            self.entries[case_id] = entry
            return
        m = self.momentum
        def ema(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            n = min(a.shape[0], b.shape[0])
            return _normalize(m * a[:n] + (1.0 - m) * b[:n])
        self.entries[case_id] = SupportEntry(
            global_feature=_normalize(m * old.global_feature + (1.0 - m) * entry.global_feature),
            foreground=ema(old.foreground, entry.foreground),
            background=ema(old.background, entry.background),
            boundary=ema(old.boundary, entry.boundary),
        )

    def retrieve(
        self, query_feature: torch.Tensor, top_k: int, temperature: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not self.entries:
            return None
        query_global = _normalize(query_feature.detach().float().flatten(2).mean(dim=2)).squeeze(0)
        names = list(self.entries)
        globals_ = torch.stack([self.entries[name].global_feature for name in names], dim=0).to(query_feature.device)
        scores = globals_ @ query_global.to(globals_.device)
        k = min(max(1, int(top_k)), len(names))
        values, indices = torch.topk(scores, k=k)
        weights = torch.softmax(values / max(float(temperature), 1e-4), dim=0)
        def combine(field: str) -> torch.Tensor:
            tensors = []
            for index in indices.tolist():
                tensors.append(getattr(self.entries[names[index]], field).to(query_feature.device))
            # Preserve all local prototypes and weight them by case similarity.
            return torch.cat([tensor * weights[i] for i, tensor in enumerate(tensors)], dim=0)
        return combine("foreground"), combine("background"), combine("boundary")

    @staticmethod
    def prior(
        feature_map: torch.Tensor,
        foreground: torch.Tensor,
        background: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        # feature_map [B,C,D,H,W], prototypes [P,C]
        features = _normalize(feature_map.permute(0, 2, 3, 4, 1))
        fg = _normalize(foreground.to(feature_map.device)).T
        bg = _normalize(background.to(feature_map.device)).T
        fg_score = torch.einsum("bdhwc,cp->bdhwp", features, fg).max(dim=-1).values
        bg_score = torch.einsum("bdhwc,cp->bdhwp", features, bg).max(dim=-1).values
        return torch.nan_to_num(torch.sigmoid((fg_score - bg_score) / max(float(temperature), 1e-4))).unsqueeze(1)
