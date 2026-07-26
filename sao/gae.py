"""Length-adaptive + skip-observation GAE for SAO."""

from __future__ import annotations

import torch


def length_adaptive_lambda(length: int, alpha: float = 1.5) -> float:
    """λ_policy = 1 - 1/(α · l)."""
    length = max(int(length), 1)
    return 1.0 - 1.0 / (alpha * length)


def gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 1.0,
    done: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard token-level GAE. rewards/values: [T]."""
    t = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = 0.0
    if done is None:
        done = torch.zeros(t, dtype=rewards.dtype, device=rewards.device)
    for i in reversed(range(t)):
        next_v = 0.0 if i == t - 1 else values[i + 1]
        nonterminal = 1.0 - done[i]
        delta = rewards[i] + gamma * next_v * nonterminal - values[i]
        last = delta + gamma * lam * nonterminal * last
        adv[i] = last
    return adv


def skip_obs_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    is_action: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 1.0,
) -> torch.Tensor:
    """GAE only along action tokens; observations are skipped in the backup.

    is_action: bool [T], True for model-generated tokens.
    Observation tokens get advantage 0 and are not used as V(s_{t+1}) bridges
    unless they are the next action start (we bridge action_end -> next_action_start).
    """
    t = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    action_idx = torch.nonzero(is_action, as_tuple=False).flatten()
    if action_idx.numel() == 0:
        return adv

    # Work on the action subsequence; rewards on obs are ignored for backup.
    a_rewards = rewards[action_idx].clone()
    a_values = values[action_idx].clone()
    # Put terminal reward on last action token if provided on last step.
    a_adv = gae_advantages(a_rewards, a_values, gamma=gamma, lam=lam)
    adv[action_idx] = a_adv
    return adv
