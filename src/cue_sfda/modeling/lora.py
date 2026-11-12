from __future__ import annotations

import math
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class LoRAMultiheadAttention(nn.Module):
    """Frozen MultiheadAttention with trainable low-rank Q/K/V deltas."""

    def __init__(
        self,
        base: nn.MultiheadAttention,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not base._qkv_same_embed_dim:
            raise ValueError("Packed QKV weights are required")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.adapter_enabled = True
        dimension = base.embed_dim
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        for letter in "qkv":
            matrix_a = nn.Parameter(
                base.in_proj_weight.new_empty((self.rank, dimension))
            )
            matrix_b = nn.Parameter(
                base.in_proj_weight.new_zeros((dimension, self.rank))
            )
            nn.init.kaiming_uniform_(matrix_a, a=math.sqrt(5))
            self.register_parameter(f"lora_{letter}_A", matrix_a)
            self.register_parameter(f"lora_{letter}_B", matrix_b)

    def _delta(self, letter: str) -> torch.Tensor:
        matrix_a = getattr(self, f"lora_{letter}_A")
        matrix_b = getattr(self, f"lora_{letter}_B")
        return matrix_b @ matrix_a * self.scaling

    def _merged_qkv(self) -> torch.Tensor:
        weight = self.base.in_proj_weight
        if not self.adapter_enabled:
            return weight
        delta = torch.cat([self._delta(letter) for letter in "qkv"], dim=0)
        delta = delta.to(device=weight.device, dtype=weight.dtype)
        return weight + delta

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        is_batched = query.dim() == 3
        if self.base.batch_first and is_batched:
            query = query.transpose(1, 0)
            key = key.transpose(1, 0)
            value = value.transpose(1, 0)
        query = self.dropout(query)
        output, weights = F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.base.embed_dim,
            num_heads=self.base.num_heads,
            in_proj_weight=self._merged_qkv(),
            in_proj_bias=self.base.in_proj_bias,
            bias_k=self.base.bias_k,
            bias_v=self.base.bias_v,
            add_zero_attn=self.base.add_zero_attn,
            dropout_p=self.base.dropout if self.training else 0.0,
            out_proj_weight=self.base.out_proj.weight,
            out_proj_bias=self.base.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if self.base.batch_first and is_batched:
            output = output.transpose(1, 0)
        return output, weights

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = super().__getattr__("base")
            return getattr(base, name)


class LoRALinear(nn.Module):
    """Frozen linear layer with a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / self.rank
        self.adapter_enabled = True
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_A = nn.Parameter(base.weight.new_empty((rank, base.in_features)))
        self.lora_B = nn.Parameter(base.weight.new_zeros((base.out_features, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.base(value)
        if not self.adapter_enabled:
            return output
        residual = F.linear(F.linear(value, self.lora_A), self.lora_B)
        return output + residual * self.scaling


@contextmanager
def lora_disabled(model: nn.Module) -> Iterator[None]:
    """Temporarily run the frozen source model without any LoRA deltas."""
    modules = [
        module for module in model.modules()
        if isinstance(module, (LoRAMultiheadAttention, LoRALinear))
    ]
    previous = [module.adapter_enabled for module in modules]
    try:
        for module in modules:
            module.adapter_enabled = False
        yield
    finally:
        for module, enabled in zip(modules, previous, strict=True):
            module.adapter_enabled = enabled


def inject_decoder_qkv_lora(
    decoder: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
) -> int:
    replaced = 0
    for module in decoder.modules():
        child = module._modules.get("multihead_attn")
        if isinstance(child, LoRAMultiheadAttention):
            continue
        if isinstance(child, nn.MultiheadAttention):
            wrapper = LoRAMultiheadAttention(child, rank, alpha, dropout)
            module.multihead_attn = wrapper
            replaced += 1
    if replaced == 0:
        raise RuntimeError("No decoder cross-attention modules found for QKV LoRA")
    return replaced


def inject_linear_lora(module: nn.Module, rank: int, alpha: float) -> int:
    replaced = 0
    # Snapshot before replacement. Iterating the live tree would descend into
    # each new wrapper's frozen ``base`` linear and wrap it recursively.
    for parent in list(module.modules()):
        for name, child in list(parent._modules.items()):
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear):
                parent._modules[name] = LoRALinear(child, rank, alpha)
                replaced += 1
    return replaced


def freeze_except_lora(model: nn.Module) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable = []
    for module in model.modules():
        if not isinstance(module, (LoRAMultiheadAttention, LoRALinear)):
            continue
        for name, parameter in module.named_parameters(recurse=False):
            if name.startswith("lora_"):
                parameter.requires_grad = True
                trainable.append(parameter)
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters were selected")
    return trainable


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if ".lora_" in name
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = set(lora_state_dict(model))
    supplied = set(state)
    if expected != supplied:
        missing = sorted(expected.difference(supplied))
        unexpected = sorted(supplied.difference(expected))
        message = f"LoRA state mismatch. Missing={missing}, unexpected={unexpected}"
        raise RuntimeError(message)
    model.load_state_dict(state, strict=False)
