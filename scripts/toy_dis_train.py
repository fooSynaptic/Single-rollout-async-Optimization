"""Toy loop: synthetic tokens, DIS on/off, K=2 critic.

No HF model — verifies DIS + GAE + K=2 wiring.

  scripts/py.sh scripts/toy_dis_train.py --out logs/toy_dis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sao.critic_loop import run_update_step
from sao.gae import gae_advantages, length_adaptive_lambda
from sao.loss import dis_policy_weights


class TinyPolicy(nn.Module):
    def __init__(self, vocab: int = 64, dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.out = nn.Linear(dim, vocab)

    def logprob(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.out(self.embed(tokens))
        logp = torch.log_softmax(logits, dim=-1)
        return logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


class TinyCritic(nn.Module):
    def __init__(self, vocab: int = 64, dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.v = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.v(self.embed(tokens)).squeeze(-1)


def mc_returns(rewards: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(rewards)
    g = 0.0
    for i in reversed(range(rewards.numel())):
        g = float(rewards[i]) + g
        out[i] = g
    return out


def run_episode(policy, critic, opt_p, opt_c, *, use_dis: bool):
    t_len = 16
    tokens = torch.randint(0, 64, (t_len,))
    with torch.no_grad():
        log_roll = policy.logprob(tokens).detach().clone()
        log_roll[3] -= 3.0  # off-policy spikes → should be masked by DIS
        log_roll[7] += 3.0

    rewards = torch.zeros(t_len)
    rewards[-1] = 1.0
    with torch.no_grad():
        values = critic(tokens)
        lam = length_adaptive_lambda(t_len, alpha=1.5)
        adv = gae_advantages(rewards, values, gamma=1.0, lam=lam)
        returns = mc_returns(rewards)

    def critic_step():
        loss = ((critic(tokens) - returns) ** 2).mean()
        opt_c.zero_grad()
        loss.backward()
        opt_c.step()
        return float(loss.detach())

    def policy_step():
        log_theta = policy.logprob(tokens)
        if use_dis:
            weight, metrics = dis_policy_weights(
                log_theta.detach(), log_roll, adv, lower=0.3, upper=5.0
            )
            loss = -(weight * log_theta).mean()
            mask_ratio = float(metrics["dis/mask_ratio"])
        else:
            loss = -(adv * log_theta).mean()
            mask_ratio = 0.0
        opt_p.zero_grad()
        loss.backward()
        opt_p.step()
        return {
            "loss": float(loss.detach()),
            "mask_ratio": mask_ratio,
            "adv_mean": float(adv.mean()),
        }

    return run_update_step(policy_step=policy_step, critic_step=critic_step, k=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    results = {}
    for use_dis, name in [(True, "with_dis"), (False, "no_dis")]:
        torch.manual_seed(args.seed)
        policy = TinyPolicy()
        critic = TinyCritic()
        opt_p = torch.optim.Adam(policy.parameters(), lr=1e-2)
        opt_c = torch.optim.Adam(critic.parameters(), lr=1e-2)
        hist = []
        for step in range(args.steps):
            out = run_episode(policy, critic, opt_p, opt_c, use_dis=use_dis)
            hist.append(
                {
                    "step": step,
                    "policy_loss": out["policy_metrics"]["loss"],
                    "mask_ratio": out["policy_metrics"]["mask_ratio"],
                    "adv_mean": out["policy_metrics"]["adv_mean"],
                    "critic_updates": out["critic_updates"],
                }
            )
        results[name] = hist
        mean_mask = sum(h["mask_ratio"] for h in hist) / len(hist)
        print(name, "final_loss", hist[-1]["policy_loss"], "mean_mask", mean_mask)

    mean_mask = sum(h["mask_ratio"] for h in results["with_dis"]) / len(results["with_dis"])
    assert mean_mask > 0.0
    assert results["with_dis"][0]["critic_updates"] == 2
    print("OK: toy DIS loop")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
