import random

import numpy as np
import pytest
import torch
from torch import nn

from cue_sfda.data import CaseRecord
from cue_sfda.engine.checkpoint import save_checkpoint
from cue_sfda.engine.trainer import CUETrainer
from cue_sfda.method.dynamic_hard import DynamicHardCache
from cue_sfda.method.uncertainty import ConfidencePartition
from cue_sfda.modeling.lora import inject_decoder_qkv_lora


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(8, 2)


class QuietLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class Views:
    def __init__(self):
        self.generator = torch.Generator().manual_seed(999)
        self.python_generator = random.Random(777)


def _cache_metadata(case_name, version):
    teacher = "frozen_source" if version == 0 else "student_snapshot"
    return {
        "schema_version": 1,
        "cache_version": version,
        "teacher_type": teacher,
        "num_cases": 1,
        "case_ids": [case_name],
        "num_views": 10,
        "prompts": ["liver cancer"],
        "standard_prompt": "liver cancer",
        "image_temperature": 0.15,
        "prompt_temperature": 0.15,
        "refresh_tile_step_size": 1.0,
        "teacher_iteration": None if version == 0 else 123,
        "source_checkpoint": None,
    }


@pytest.mark.parametrize(
    ("refresh_cycle", "seen", "history", "expected_version"),
    [
        (0, ["case.nii.gz"], [], 0),
        (
            1,
            [],
            [
                {
                    "refresh_cycle": 1,
                    "trigger_iteration": 123,
                    "covered_cases": 1,
                    "teacher_type": "student_snapshot",
                }
            ],
            1,
        ),
    ],
)
def test_resume_restores_coverage_rng_refresh_count_and_cache_version(
    tmp_path, refresh_cycle, seen, history, expected_version
):
    torch.manual_seed(117)
    random.seed(117)
    np.random.seed(117)
    model = ToyModel()
    inject_decoder_qkv_lora(model, rank=2, alpha=2.0, dropout=0.0)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    trainer_generator = torch.Generator().manual_seed(311)
    views = Views()
    image_path = tmp_path / "case.nii.gz"
    image_path.write_bytes(b"source image")
    case = CaseRecord(name=image_path.name, image_path=image_path)
    cache = DynamicHardCache(tmp_path / "cache")
    stat = image_path.stat()
    for version, value in ((0, 0.2), (1, 0.8)):
        probability = np.full((1, 2, 2, 2), value, dtype=np.float32)
        confidence = np.full_like(probability, 0.5)
        cache.save_case(
            case,
            version,
            probability,
            confidence,
            {
                "case_id": case.name,
                "image_size": int(stat.st_size),
                "image_mtime_ns": int(stat.st_mtime_ns),
                "padded_shape": [2, 2, 2],
                "cache_version": version,
                "teacher_type": (
                    "frozen_source" if version == 0 else "student_snapshot"
                ),
                    "teacher_iteration": None if version == 0 else 123,
            },
        )
        cache.write_version_metadata(version, _cache_metadata(case.name, version))
    # Simulate a crash after another cache version was activated; the
    # checkpoint state must select its own version during resume.
    cache.activate_version(1 - expected_version)

    trainer_state = {
        "dynamic_hard_enabled": True,
        "refresh_cycle": refresh_cycle,
        "seen_cases_since_refresh": seen,
        "refresh_history": history,
        "branch_sample_counts": {"high": 10, "low": 20},
        "branch_patch_counts": {"high": 30, "low": 60},
        "branch_effective_patch_counts": {"high": 29, "low": 58},
        "branch_skipped_patch_counts": {"high": 1, "low": 2},
    }
    checkpoint_path = tmp_path / "latest.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        iteration=150 if refresh_cycle == 0 else 200,
        config={},
        confidence_scores={case.name: 0.0},
        confidence_threshold=0.5,
        trainer_generator_state=trainer_generator.get_state(),
        view_generator_state=views.generator.get_state(),
        view_python_generator_state=views.python_generator.getstate(),
        trainer_state=trainer_state,
    )

    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_cpu = torch.rand(3)
    expected_trainer_rng = torch.rand(3, generator=trainer_generator)
    expected_view_rng = torch.rand(3, generator=views.generator)
    expected_view_python = views.python_generator.random()

    # Perturb all RNG streams before exercising the real trainer resume path.
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    trainer_generator.manual_seed(9)
    views.generator.manual_seed(9)
    views.python_generator.seed(9)

    resumed_model = ToyModel()
    inject_decoder_qkv_lora(resumed_model, rank=2, alpha=2.0, dropout=0.0)
    resumed_optimizer = torch.optim.AdamW(
        [parameter for parameter in resumed_model.parameters() if parameter.requires_grad],
        lr=9e-4,
    )
    trainer = object.__new__(CUETrainer)
    trainer.model = resumed_model
    trainer.optimizer = resumed_optimizer
    trainer.scaler = None
    trainer.config = {"train": {"learning_rate": 1e-4}}
    trainer.dynamic_hard_enabled = True
    trainer.output_dir = tmp_path
    trainer.logger = QuietLogger()
    trainer.generator = torch.Generator().manual_seed(9)
    trainer.views = Views()
    trainer.partition = ConfidencePartition(0.5, 0.9)
    trainer.branch_sample_counts = {"high": 0, "low": 0}
    trainer.branch_patch_counts = {"high": 0, "low": 0}
    trainer.branch_effective_patch_counts = {"high": 0, "low": 0}
    trainer.branch_skipped_patch_counts = {"high": 0, "low": 0}
    trainer.dynamic_hard_loss_total = 0.0
    trainer.dynamic_hard_loss_patches = 0
    trainer._resume(checkpoint_path)

    assert trainer.refresh_cycle == refresh_cycle
    assert trainer.seen_cases_since_refresh == set(seen)
    assert trainer.refresh_history == history
    assert trainer.branch_sample_counts == {"high": 10, "low": 20}
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    torch.testing.assert_close(torch.rand(3), expected_cpu)
    torch.testing.assert_close(
        torch.rand(3, generator=trainer.generator), expected_trainer_rng
    )
    torch.testing.assert_close(
        torch.rand(3, generator=trainer.views.generator), expected_view_rng
    )
    assert trainer.views.python_generator.random() == expected_view_python

    trainer.dynamic_hard_enabled = True
    trainer.dynamic_hard_cache = cache
    trainer.dynamic_hard_config = {
        "prompts": ["liver cancer"],
        "standard_prompt": "liver cancer",
        "image_temperature": 0.15,
        "prompt_temperature": 0.15,
        "refresh_tile_step_size": 1.0,
    }
    trainer.config = {
        "method": {"num_views": 10},
        "model": {},
        "dynamic_hard": trainer.dynamic_hard_config,
    }
    trainer.train_cases = [case]
    trainer.max_refresh_cycles = 1
    trainer.output_dir = tmp_path / "resume_output"
    trainer.logger = QuietLogger()
    trainer._ensure_dynamic_hard_cache()
    assert cache.metadata()["active_version"] == expected_version
    assert cache.metadata()["teacher_type"] == (
        "frozen_source" if expected_version == 0 else "student_snapshot"
    )
    expected_probability = 0.2 if expected_version == 0 else 0.8
    np.testing.assert_allclose(
        cache.load_case(case, expected_version)["joint_probability"],
        expected_probability,
    )
