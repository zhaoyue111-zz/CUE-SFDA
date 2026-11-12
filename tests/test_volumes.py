from pathlib import Path

import torch

from cue_sfda.data.volumes import CaseRecord, TargetVolumeStore


def test_sliding_window_covers_padded_volume():
    store = TargetVolumeStore((32, 32, 32))
    case = CaseRecord("case.nii.gz", Path("case.nii.gz"))
    source = torch.zeros((1, 20, 64, 80))
    store.load = lambda _: source
    volume, slicers = store.sliding_window(case, step_size=0.5)
    coverage = torch.zeros(volume.shape[-3:], dtype=torch.int32)
    for slicer in slicers:
        patch = store.extract_patch(volume, slicer)
        assert patch.shape == (1, 32, 32, 32)
        coverage[slicer] += 1
    assert volume.shape == (1, 32, 64, 80)
    assert int(coverage.min()) >= 1
