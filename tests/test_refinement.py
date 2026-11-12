import torch

from cue_sfda.method.refinement import affinity_random_walk


def test_random_walk_preserves_constant_probability():
    probability = torch.full((1, 1, 8, 8, 8), 0.7)
    features = torch.ones((1, 4, 8, 8, 8))
    refined = affinity_random_walk(
        probability,
        features,
        output_size=(8, 8, 8),
        graph_size=(4, 4, 4),
        steps=3,
        alpha=0.6,
        temperature=0.15,
    )
    assert refined.shape == probability.shape
    assert torch.allclose(refined, probability, atol=1e-5)


def test_random_walk_output_is_bounded():
    probability = torch.rand((1, 1, 8, 8, 8))
    features = torch.rand((1, 3, 8, 8, 8))
    refined = affinity_random_walk(
        probability,
        features,
        output_size=(8, 8, 8),
        graph_size=(4, 4, 4),
        steps=2,
        alpha=0.5,
        temperature=0.2,
    )
    assert float(refined.min()) >= 0.0
    assert float(refined.max()) <= 1.0


def test_max_score_pooling_preserves_a_small_seed():
    probability = torch.zeros((1, 1, 8, 8, 8))
    probability[..., 3, 3, 3] = 1.0
    features = torch.ones((1, 2, 8, 8, 8))
    refined = affinity_random_walk(
        probability,
        features,
        output_size=(8, 8, 8),
        graph_size=(4, 4, 4),
        steps=1,
        alpha=1.0,
        temperature=0.2,
        score_pooling="max",
    )
    assert float(refined.max()) > 0.1
