import torch
from torch import nn

from cue_sfda.modeling.lora import (
    LoRALinear,
    LoRAMultiheadAttention,
    freeze_except_lora,
    inject_decoder_qkv_lora,
    inject_linear_lora,
    lora_disabled,
)


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(16, 4)
        self.multihead_attn = nn.MultiheadAttention(16, 4)


def test_lora_initialization_preserves_attention_output():
    torch.manual_seed(3)
    base = nn.MultiheadAttention(16, 4, dropout=0.0).eval()
    query = torch.randn(5, 2, 16)
    expected, _ = base(query, query, query)
    wrapped = LoRAMultiheadAttention(base, rank=2, alpha=4.0).eval()
    actual, _ = wrapped(query, query, query)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_injection_selects_only_lora_parameters():
    decoder = ToyDecoder()
    count = inject_decoder_qkv_lora(decoder, rank=2, alpha=4.0, dropout=0.0)
    trainable = freeze_except_lora(decoder)
    assert count == 1
    assert isinstance(decoder.self_attn, nn.MultiheadAttention)
    assert isinstance(decoder.multihead_attn, LoRAMultiheadAttention)
    assert len(trainable) == 6
    assert sum(parameter.numel() for parameter in trainable) == 192
    assert all(parameter.requires_grad for parameter in trainable)


def test_lora_can_be_disabled_for_frozen_source_teacher():
    torch.manual_seed(7)
    base = nn.MultiheadAttention(16, 4, dropout=0.0).eval()
    query = torch.randn(5, 2, 16)
    expected, _ = base(query, query, query)
    wrapped = LoRAMultiheadAttention(base, rank=2, alpha=4.0).eval()
    with torch.no_grad():
        wrapped.lora_q_B.fill_(0.2)
    adapted, _ = wrapped(query, query, query)
    with lora_disabled(wrapped):
        source, _ = wrapped(query, query, query)
    restored, _ = wrapped(query, query, query)
    assert not torch.allclose(adapted, expected)
    assert torch.allclose(source, expected, atol=1e-6)
    assert torch.allclose(restored, adapted, atol=1e-6)


def test_linear_lora_injection_uses_a_frozen_source_path():
    torch.manual_seed(11)
    module = nn.Sequential(nn.Linear(8, 12), nn.GELU(), nn.Linear(12, 3)).eval()
    value = torch.randn(4, 8)
    expected = module(value)
    count = inject_linear_lora(module, rank=2, alpha=4.0)
    assert count == 2
    assert isinstance(module[0], LoRALinear)
    assert isinstance(module[2], LoRALinear)
    assert torch.allclose(module(value), expected, atol=1e-6)
    with torch.no_grad():
        module[0].lora_B.fill_(0.1)
    adapted = module(value)
    with lora_disabled(module):
        source = module(value)
    assert not torch.allclose(adapted, expected)
    assert torch.allclose(source, expected, atol=1e-6)
