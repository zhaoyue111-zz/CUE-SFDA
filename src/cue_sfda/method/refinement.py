from __future__ import annotations

import torch
import torch.nn.functional as F


def affinity_random_walk(
    probability: torch.Tensor,
    features: torch.Tensor,
    output_size: tuple[int, int, int],
    graph_size: tuple[int, int, int],
    steps: int,
    alpha: float,
    temperature: float,
    score_pooling: str = "average",
    image: torch.Tensor | None = None,
) -> torch.Tensor:
    if probability.ndim != 5 or features.ndim != 5:
        raise ValueError("Random-walk inputs must be five-dimensional")
    anchor_strength = float(alpha)
    if not 0.0 <= anchor_strength <= 1.0:
        raise ValueError("random-walk alpha must be in [0, 1]")
    if score_pooling == "max":
        score = F.adaptive_max_pool3d(probability, graph_size).float()
    elif score_pooling == "average":
        score = F.adaptive_avg_pool3d(probability, graph_size).float()
    else:
        raise ValueError(f"Unknown score pooling mode: {score_pooling}")
    anchor = score
    feature = F.adaptive_avg_pool3d(features, graph_size).float()
    pixels = None
    if image is not None:
        pixels = F.adaptive_avg_pool3d(image, graph_size).float()
    for _ in range(int(steps)):
        propagated = torch.zeros_like(score)
        normalizer = torch.zeros_like(score)
        for offset_d in (-1, 0, 1):
            for offset_h in (-1, 0, 1):
                for offset_w in (-1, 0, 1):
                    offset = (offset_d, offset_h, offset_w)
                    if offset == (0, 0, 0):
                        continue
                    neighbor_score = _shift_3d(score, offset)
                    neighbor_feature = _shift_3d(feature, offset)
                    valid = _valid_mask_3d(score, offset)
                    difference = (feature - neighbor_feature).abs().mean(
                        dim=1, keepdim=True
                    )
                    feature_affinity = torch.exp(
                        -difference / max(float(temperature), 1e-6)
                    )
                    if pixels is None:
                        weight = feature_affinity
                    else:
                        neighbor_pixels = _shift_3d(pixels, offset)
                        pixel_affinity = torch.exp(
                            -(pixels - neighbor_pixels).abs().mean(
                                dim=1, keepdim=True
                            )
                        )
                        # Eq. (18): I = I_f * I_x + I_f.
                        weight = feature_affinity * (1.0 + pixel_affinity)
                    weight = weight * valid
                    propagated = propagated + weight * neighbor_score
                    normalizer = normalizer + weight
        propagated = propagated / normalizer.clamp_min(1e-6)
        # Liver organ experiments can retain part of the initial teacher score
        # while propagating. alpha=0 preserves the original unanchored update.
        score = (1.0 - anchor_strength) * propagated + anchor_strength * anchor
    refined = F.interpolate(
        score, size=output_size, mode="trilinear", align_corners=False
    )
    return refined.clamp(0.0, 1.0)


def _shift_3d(tensor: torch.Tensor, offset: tuple[int, int, int]) -> torch.Tensor:
    shifted = torch.roll(tensor, shifts=offset, dims=(2, 3, 4))
    for axis, direction in zip((2, 3, 4), offset, strict=True):
        if direction == 0:
            continue
        boundary = [slice(None)] * tensor.ndim
        boundary[axis] = 0 if direction > 0 else -1
        shifted[tuple(boundary)] = 0
    return shifted


def _valid_mask_3d(
    tensor: torch.Tensor, offset: tuple[int, int, int]
) -> torch.Tensor:
    mask = torch.ones_like(tensor[:, :1])
    for axis, direction in zip((2, 3, 4), offset, strict=True):
        if direction == 0:
            continue
        boundary = [slice(None)] * mask.ndim
        boundary[axis] = 0 if direction > 0 else -1
        mask[tuple(boundary)] = 0
    return mask
