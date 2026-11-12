import torch
from torch import nn

from cue_sfda.modeling.voxtell import DynamicVoxTell


class FakeSourceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, image, text_embedding, return_features=False):
        self.calls += 1
        prediction = image[:, :1] + text_embedding.mean() * 0.0
        if not return_features:
            return prediction
        feature = image[:, :1].unsqueeze(1)
        return prediction, {"d5_pre_head": feature}


def test_native_patch_delegates_to_original_voxtell_forward():
    source = FakeSourceModel()
    model = DynamicVoxTell(source, native_patch_size=(4, 4, 4))
    image = torch.randn(1, 1, 4, 4, 4)
    embedding = torch.randn(1, 1, 8)
    prediction = model(image, embedding)
    prediction_with_features, features = model(
        image,
        embedding,
        return_features=True,
    )
    assert source.calls == 2
    assert torch.equal(prediction, image)
    assert torch.equal(prediction_with_features, image)
    assert torch.equal(features, image)
