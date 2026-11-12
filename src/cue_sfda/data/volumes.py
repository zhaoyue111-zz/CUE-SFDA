from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.inference.sliding_window_prediction import (
    compute_steps_for_sliding_window,
)
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import (
    ZScoreNormalization,
)


@dataclass(frozen=True)
class CaseRecord:
    name: str
    image_path: Path


def load_split(
    image_dir: str | Path,
    split_json: str | Path,
) -> tuple[list[CaseRecord], list[CaseRecord]]:
    image_root = Path(image_dir)
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_root}")
    with Path(split_json).open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    train_entries = split.get("train_cases", split.get("train"))
    test_entries = split.get("test_cases", split.get("test"))
    if train_entries is None or test_entries is None:
        raise KeyError("Split JSON must contain train_cases and test_cases")

    def build(entries: list[str], split_name: str) -> list[CaseRecord]:
        records = []
        for entry in entries:
            name = Path(entry).name
            image_path = image_root / name
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing {split_name} image: {image_path}")
            records.append(CaseRecord(name=name, image_path=image_path))
        return records

    train_cases = build(train_entries, "train")
    test_cases = build(test_entries, "test")
    overlap = {case.name for case in train_cases}.intersection(
        case.name for case in test_cases
    )
    if overlap:
        raise ValueError(f"Train/test overlap: {sorted(overlap)[:5]}")
    return train_cases, test_cases


class TargetVolumeStore:
    def __init__(self, patch_size: tuple[int, int, int], cache_size: int = 8) -> None:
        self.patch_size = tuple(int(value) for value in patch_size)
        self.cache_size = int(cache_size)
        self.reader = NibabelIOWithReorient()
        self.normalizer = ZScoreNormalization(intensityproperties={})
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def load(self, case: CaseRecord) -> torch.Tensor:
        cached = self.cache.pop(case.name, None)
        if cached is not None:
            self.cache[case.name] = cached
            return cached
        image, _ = self.reader.read_images([str(case.image_path)])
        image = image.astype(np.float32, copy=False)
        image, _, _ = crop_to_nonzero(image, None)
        image = self.normalizer.run(image, None)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
        self.cache[case.name] = tensor
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return tensor

    def sliding_window(
        self,
        case: CaseRecord,
        step_size: float,
    ) -> tuple[torch.Tensor, list[tuple[slice, slice, slice]]]:
        volume, _ = pad_nd_image(
            self.load(case),
            self.patch_size,
            "constant",
            {"value": 0},
            True,
            None,
        )
        spatial_shape = tuple(int(value) for value in volume.shape[-3:])
        steps = compute_steps_for_sliding_window(
            spatial_shape,
            self.patch_size,
            float(step_size),
        )
        slicers = []
        for start_d in steps[0]:
            for start_h in steps[1]:
                for start_w in steps[2]:
                    slicers.append(
                        (
                            slice(start_d, start_d + self.patch_size[0]),
                            slice(start_h, start_h + self.patch_size[1]),
                            slice(start_w, start_w + self.patch_size[2]),
                        )
                    )
        return volume, slicers

    @staticmethod
    def extract_patch(
        volume: torch.Tensor,
        slicer: tuple[slice, slice, slice],
    ) -> torch.Tensor:
        return volume[(slice(None), *slicer)].clone()
