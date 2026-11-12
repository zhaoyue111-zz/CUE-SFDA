import torch
from torch import nn

from cue_sfda.engine.checkpoint import save_checkpoint
from cue_sfda.modeling.lora import inject_decoder_qkv_lora


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(8, 2)


def test_checkpoint_records_best_validation_metadata(tmp_path):
    model = ToyModel()
    inject_decoder_qkv_lora(model, rank=2, alpha=2.0, dropout=0.0)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    path = tmp_path / "latest.pt"
    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        iteration=20,
        config={"experiment": {"gpu_id": 1}},
        confidence_scores={"case": 0.8},
        confidence_threshold=0.5,
        best_dice=0.42,
        best_iteration=20,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["iteration"] == 20
    assert payload["best_dice"] == 0.42
    assert payload["best_iteration"] == 20
    assert payload["format_version"] == 3
    assert "rng_state" in payload
    assert payload["rng_state"]["torch_cpu"] is not None
