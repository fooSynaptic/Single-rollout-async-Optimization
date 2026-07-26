"""Critic update schedule: K value steps per 1 policy step."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def run_update_step(
    *,
    policy_step: Callable[[], Any],
    critic_step: Callable[[], Any],
    k: int = 2,
) -> dict:
    """One outer step: critic K times, then policy once.

    Returns counts for tests / logging.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    critic_metrics = []
    for _ in range(k):
        critic_metrics.append(critic_step())
    policy_metrics = policy_step()
    return {
        "critic_updates": k,
        "policy_updates": 1,
        "critic_metrics": critic_metrics,
        "policy_metrics": policy_metrics,
    }
