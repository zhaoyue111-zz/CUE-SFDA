from __future__ import annotations


def should_select_high(high_confidence: bool, mode: str) -> bool:
    return bool(high_confidence and mode != "none")


def conditional_high_step(optimizer, *, accepted: bool, effective: bool) -> bool:
    """Apply a selected High step; rejection touches neither optimizer nor scaler."""
    if not accepted or not effective:
        return False
    optimizer.step()
    return True
