import torch

from cue_sfda.method.losses import (
    balanced_consistency_loss,
    balanced_masked_binary_cross_entropy,
    bernoulli_kl_divergence,
    confidence_weight_map,
    consistency_loss,
    masked_binary_cross_entropy,
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


def test_linear_confidence_weights_keep_low_confidence_voxels():
    confidence = torch.tensor([0.0, 0.5, 1.0])
    weights = confidence_weight_map(confidence)
    assert torch.allclose(weights, torch.tensor([0.3, 0.65, 1.0]))
    assert bool((weights > 0).all())
    power_weights = confidence_weight_map(confidence, function="power", power=2.0)
    assert torch.allclose(power_weights, torch.tensor([0.3, 0.475, 1.0]))


def test_unit_confidence_weights_reproduce_unweighted_losses_and_have_gradients():
    probability = torch.tensor([0.2, 0.8, 0.7, 0.1]).reshape(1, 1, 1, 2, 2)
    probability.requires_grad_()
    target = torch.tensor([0.0, 1.0, 1.0, 0.0]).reshape_as(probability)
    unit_weight = confidence_weight_map(torch.ones_like(target))
    plain_weight = torch.ones_like(target)
    assert torch.equal(unit_weight, plain_weight)
    assert torch.equal(
        masked_binary_cross_entropy(probability, target, unit_weight),
        masked_binary_cross_entropy(probability, target, plain_weight),
    )
    assert torch.equal(
        balanced_masked_binary_cross_entropy(probability, target, unit_weight),
        balanced_masked_binary_cross_entropy(probability, target, plain_weight),
    )
    assert torch.equal(
        masked_soft_dice_loss(probability, target, unit_weight),
        masked_soft_dice_loss(probability, target, plain_weight),
    )
    masked_soft_dice_loss(probability, target, unit_weight).backward()
    assert probability.grad is not None and bool(torch.isfinite(probability.grad).all())
    assert bool((probability.grad != 0).any())
    probability.grad = None
    confidence_weight = confidence_weight_map(torch.tensor([0.1, 0.4, 0.8, 0.3]).reshape_as(target))
    weighted_loss = balanced_masked_binary_cross_entropy(
        probability, target, confidence_weight
    ) + masked_soft_dice_loss(probability, target, confidence_weight)
    weighted_loss.backward()
    assert probability.grad is not None and bool(torch.isfinite(probability.grad).all())
    assert bool((probability.grad != 0).any())
