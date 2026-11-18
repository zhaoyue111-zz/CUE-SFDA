from __future__ import annotations


def decide(semantic_accepted: bool, update_accepted: bool) -> dict:
    accepted = bool(semantic_accepted and update_accepted)
    return {
        "accepted": accepted,
        "reason": "semantic_and_update_pass" if accepted else "semantic_or_update_condition_failed",
        "semantic_accepted": bool(semantic_accepted),
        "update_accepted": bool(update_accepted),
        "final_accepted": accepted,
    }
