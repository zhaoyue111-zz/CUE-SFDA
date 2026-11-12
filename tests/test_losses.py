import torch

from cue_sfda.method.losses import (
    balanced_consistency_loss,
    balanced_masked_binary_cross_entropy,
    bernoulli_kl_divergence,
    consistency_loss,
    masked_soft_dice_loss,
)


def test_bernoulli_kl_is_zero_for_matching_probabilities():
    probability = torch.tensor([0.1, 0.4, 0.8])
    assert torch.allclose(
        bernoulli_kl_divergence(probability, probability), torch.tensor(0.0)
    )


def test_foreground_weighted_consistency_emphasizes_teacher_positive_voxels():
    teacher = torch.tensor([0.0, 1.0]).reshape(1, 1, 1, 1, 2)
    student = torch.tensor([0.0, 0.0]).reshape(1, 1, 1, 1, 2)
    unweighted = consistency_loss(student, teacher)
    weighted = consistency_loss(student, teacher, foreground_weight=5.0)
    assert weighted > unweighted


def test_weighted_soft_dice_remains_bounded_with_weights_above_one():
    probability = torch.tensor([0.2, 0.8]).reshape(1, 1, 1, 1, 2)
    target = torch.tensor([0.0, 1.0]).reshape(1, 1, 1, 1, 2)
    weight = torch.tensor([1.0, 5.0]).reshape(1, 1, 1, 1, 2)
    loss = masked_soft_dice_loss(probability, target, weight)
    assert 0.0 <= float(loss) <= 1.0


def test_balanced_bce_does_not_let_background_count_dominate():
    target = torch.zeros((1, 1, 1, 1, 101))
    target[..., 0] = 1.0
    probability = target.clone()
    probability[..., 0] = 0.5
    weight = torch.ones_like(target)
    loss = balanced_masked_binary_cross_entropy(probability, target, weight)
    expected = -torch.log(torch.tensor(0.5)) / 2
    assert torch.allclose(loss, expected, atol=1e-5)


def test_balanced_consistency_keeps_sparse_foreground_signal():
    teacher = torch.zeros((1, 1, 1, 1, 101))
    teacher[..., 0] = 1.0
    student = torch.zeros_like(teacher)
    assert torch.allclose(balanced_consistency_loss(student, teacher), torch.tensor(0.5))
