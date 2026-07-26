"""DIS loss weights for token-level policy gradient."""

from __future__ import annotations

import torch

from .dis import dis_mask, dis_token_weight, importance_ratio, mask_ratio


def dis_policy_weights(
    log_pi_theta: torch.Tensor,
    log_pi_rollout: torch.Tensor,
    advantages: torch.Tensor,
    lower: float = 0.3,
    upper: float = 5.0,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Return per-token loss weight = f(r) * A, plus metrics.

    Loss form (paper eq.1 style): E[ f(r,ε) * A * log π_θ ].
    Outside trust region, f=0 → token drops out of gradient.
    """
    ratio = importance_ratio(log_pi_theta, log_pi_rollout)
    f = dis_token_weight(ratio, lower, upper)
    keep = dis_mask(ratio, lower, upper)
    if valid is None:
        valid = torch.ones_like(keep)
    valid = valid.bool()
    weight = f * advantages
    weight = torch.where(valid, weight, torch.zeros_like(weight))
    metrics = {
        "dis/mask_ratio": mask_ratio(keep, valid).detach(),
        "dis/ratio_mean": ratio[valid].mean().detach() if valid.any() else torch.tensor(0.0),
    }
    return weight, metrics
