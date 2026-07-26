from __future__ import annotations

import torch

from sao.critic_loop import run_update_step
from sao.gae import length_adaptive_lambda, skip_obs_gae


def test_skip_obs_gae_ignores_observation_tokens() -> None:
    rewards = torch.tensor([0.0, 100.0, 1.0])
    values = torch.zeros(3)
    is_action = torch.tensor([True, False, True])
    advantages = skip_obs_gae(rewards, values, is_action)
    assert advantages.tolist() == [1.0, 0.0, 1.0]


def test_length_adaptive_lambda() -> None:
    assert length_adaptive_lambda(1, alpha=2.0) == 0.5
    assert length_adaptive_lambda(10, alpha=2.0) == 0.95


def test_critic_k_updates_before_policy() -> None:
    calls: list[str] = []
    result = run_update_step(
        critic_step=lambda: calls.append("critic"),
        policy_step=lambda: calls.append("policy"),
        k=2,
    )
    assert calls == ["critic", "critic", "policy"]
    assert result["critic_updates"] == 2
    assert result["policy_updates"] == 1
