"""SAO DIS helpers: ratio from rollout logprobs + double-sided mask."""

from __future__ import annotations

import torch


def importance_ratio(
    log_pi_theta: torch.Tensor,
    log_pi_rollout: torch.Tensor,
) -> torch.Tensor:
    """r_t = exp(log π_θ − log π_rollout)."""
    return torch.exp(log_pi_theta - log_pi_rollout)


def dis_token_weight(
    ratio: torch.Tensor,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Keep the importance ratio inside ``[lower, upper]``; mask it otherwise.

    Unlike PPO clip, out-of-range tokens contribute zero weight, not clipped r.
    """
    inside = (ratio >= lower) & (ratio <= upper)
    return torch.where(inside, ratio, torch.zeros_like(ratio))


def dis_mask(
    ratio: torch.Tensor,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Boolean mask: True = token kept for policy gradient."""
    return (ratio >= lower) & (ratio <= upper)


def mask_ratio(mask: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Fraction of valid tokens that were masked out (outside trust region)."""
    if valid is None:
        valid = torch.ones_like(mask, dtype=torch.bool)
    valid = valid.bool()
    n = valid.sum().clamp_min(1)
    masked_out = (~mask) & valid
    return masked_out.sum().to(torch.float32) / n.to(torch.float32)
