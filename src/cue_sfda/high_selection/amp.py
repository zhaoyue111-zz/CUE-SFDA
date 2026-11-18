from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def gradients_are_finite(parameters: Iterable[torch.nn.Parameter]) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


def reset_scaler_after_rejection(scaler, previous_scale: float) -> None:
    """Clear GradScaler's per-optimizer UNSCALED stage without changing saved state."""
    scaler.update(new_scale=float(previous_scale))


def clipped_gradients_are_finite(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> bool:
    parameters = list(parameters)
    norm = torch.nn.utils.clip_grad_norm_(parameters, float(max_norm))
    norm_value = float(norm.detach().float().item()) if torch.is_tensor(norm) else float(norm)
    return math.isfinite(norm_value) and gradients_are_finite(parameters)


def scaler_will_skip_step(scaler, optimizer) -> bool:
    """Inspect GradScaler's unscale result before update clears its per-optimizer state."""
    if not scaler.is_enabled():
        return False
    states = getattr(scaler, "_per_optimizer_states", None)
    if states is None or id(optimizer) not in states:
        raise RuntimeError("GradScaler has no unscale state for the selected optimizer")
    found_inf = states[id(optimizer)].get("found_inf_per_device", {})
    if not found_inf:
        raise RuntimeError("GradScaler overflow state is unavailable after unscale_")
    return any(float(value.detach().item()) != 0.0 for value in found_inf.values())


def conditional_scaler_step(
    scaler,
    optimizer,
    parameters: Iterable[torch.nn.Parameter],
    *,
    accepted: bool,
    effective: bool,
    overflow: bool,
    max_norm: float,
    scale_before: float,
) -> bool:
    """Finish one already-unscaled High attempt using GradScaler's normal protocol.

    A selection rejection resets only transient scaler bookkeeping with the same
    scale, preserving ``state_dict``. A detected AMP overflow follows GradScaler's
    normal step/update backoff path; the scaler skips optimizer.step itself.
    """
    parameters = list(parameters)
    if overflow:
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return False
    if not accepted or not effective:
        reset_scaler_after_rejection(scaler, scale_before)
        optimizer.zero_grad(set_to_none=True)
        return False
    if not clipped_gradients_are_finite(parameters, max_norm):
        reset_scaler_after_rejection(scaler, scale_before)
        optimizer.zero_grad(set_to_none=True)
        return False
    will_skip = scaler_will_skip_step(scaler, optimizer)
    scaler.step(optimizer)
    scaler.update()
    if will_skip:
        optimizer.zero_grad(set_to_none=True)
    return not will_skip
