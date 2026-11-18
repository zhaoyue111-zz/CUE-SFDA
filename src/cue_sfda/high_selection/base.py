from __future__ import annotations


def should_select_high(high_confidence: bool, mode: str) -> bool:
    return bool(high_confidence and mode != "none")
