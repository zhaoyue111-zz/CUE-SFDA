from __future__ import annotations

from dataclasses import dataclass
import random

import torch
import torch.nn.functional as F


@dataclass
class AugmentedView:
    image: torch.Tensor
    flipped_axes: tuple[int, ...]

    def align(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.flipped_axes:
            return tensor
        return torch.flip(tensor, dims=self.flipped_axes)


class StrongViewGenerator:
    def __init__(self, config: dict, seed: int) -> None:
        self.num_views = int(config["num_views"])
        self.style_mix_probability = float(config["style_mix_probability"])
        self.noise_std = float(config["noise_std"])
        self.gamma_range = tuple(float(value) for value in config["gamma_range"])
        self.generator = torch.Generator().manual_seed(int(seed))
        self.python_generator = random.Random(int(seed))
        self.mixup_alpha = float(config.get("mixup_alpha", 0.75))
        self.blur_probability = float(config.get("blur_probability", 0.5))
        self.brightness_range = tuple(
            float(value) for value in config.get("brightness_range", [0.8, 1.2])
        )

    def make_views(
        self,
        image: torch.Tensor,
        partner: torch.Tensor,
        count: int | None = None,
    ) -> list[AugmentedView]:
        requested = self.num_views if count is None else int(count)
        views = [AugmentedView(image=image, flipped_axes=())]
        while len(views) < requested:
            views.append(self._strong_view(image, partner))
        return views

    def make_uncertainty_views(self, image: torch.Tensor) -> list[AugmentedView]:
        """Generate the K mixed strong views used by Eqs. (1)--(4).

        Spatial transforms are intentionally excluded: the paper computes
        pixel-wise statistics directly across views, and padded 3-D volumes
        make interpolation-based inverse transforms unnecessarily ambiguous.
        """
        augmented = [self._photometric_view(image) for _ in range(self.num_views)]
        mixed = []
        for _ in range(self.num_views):
            first = self.python_generator.randrange(self.num_views)
            second = self.python_generator.randrange(self.num_views - 1)
            if second >= first:
                second += 1
            mixing = self.python_generator.betavariate(
                self.mixup_alpha, self.mixup_alpha
            )
            value = mixing * augmented[first] + (1.0 - mixing) * augmented[second]
            mixed.append(AugmentedView(image=value, flipped_axes=()))
        return mixed

    def _photometric_view(self, image: torch.Tensor) -> torch.Tensor:
        valid = image != 0
        output = self._gamma(image)
        brightness_low, brightness_high = self.brightness_range
        brightness = brightness_low + (brightness_high - brightness_low) * self._rand()
        output = output * brightness
        if self._rand() < self.blur_probability:
            output = F.avg_pool3d(output, kernel_size=3, stride=1, padding=1)
        noise = torch.randn(output.shape, generator=self.generator, dtype=output.dtype)
        output = output + noise.to(output.device) * self.noise_std
        return torch.where(valid, output, torch.zeros_like(output))

    def _strong_view(self, image: torch.Tensor, partner: torch.Tensor) -> AugmentedView:
        valid = image != 0
        output = image.clone()
        if self._rand() < self.style_mix_probability:
            output = self._mix_style(output, partner)
        output = self._gamma(output)
        noise = torch.randn(output.shape, generator=self.generator, dtype=output.dtype)
        output = output + noise.to(output.device) * self.noise_std
        flipped_axes = []
        for axis in range(2, output.ndim):
            if self._rand() < 0.5:
                output = torch.flip(output, dims=(axis,))
                flipped_axes.append(axis)
        output = torch.where(valid, output, torch.zeros_like(output))
        return AugmentedView(image=output, flipped_axes=tuple(flipped_axes))

    def _mix_style(self, image: torch.Tensor, partner: torch.Tensor) -> torch.Tensor:
        spatial_dims = tuple(range(2, image.ndim))
        image_mean = image.mean(dim=spatial_dims, keepdim=True)
        image_std = image.std(dim=spatial_dims, keepdim=True, unbiased=False).clamp_min(
            1e-5
        )
        partner_mean = partner.mean(dim=spatial_dims, keepdim=True)
        partner_std = partner.std(
            dim=spatial_dims, keepdim=True, unbiased=False
        ).clamp_min(1e-5)
        mixing = 0.25 + 0.5 * self._rand()
        target_mean = mixing * image_mean + (1.0 - mixing) * partner_mean
        target_std = mixing * image_std + (1.0 - mixing) * partner_std
        return (image - image_mean) / image_std * target_std + target_mean

    def _gamma(self, image: torch.Tensor) -> torch.Tensor:
        minimum = image.amin(dim=(2, 3, 4), keepdim=True)
        maximum = image.amax(dim=(2, 3, 4), keepdim=True)
        normalized = (image - minimum) / (maximum - minimum).clamp_min(1e-5)
        low, high = self.gamma_range
        gamma = low + (high - low) * self._rand()
        return normalized.clamp(0.0, 1.0).pow(gamma) * (maximum - minimum) + minimum

    def _rand(self) -> float:
        return float(torch.rand((), generator=self.generator).item())
