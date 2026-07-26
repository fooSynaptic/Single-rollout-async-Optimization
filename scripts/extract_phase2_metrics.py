#!/usr/bin/env python3
"""Extract windowed Phase2 compare metrics from AReaL ascii stats logs."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
NUM = re.compile(r"^[+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?$")

KEYS = {
    "pg_loss": [
        "ppo_actor/update/pg_loss",
        "ppo_actor/update/actor_loss/avg",
    ],
    "value_loss": [
        "ppo_critic/update/value_loss",
        "ppo_critic/update/critic_loss/avg",
        "ppo_critic/value_loss",
        "ppo_critic/critic_loss/avg",
    ],
    "mask_ratio": [
        "ppo_actor/update/dis/mask_ratio",
        "ppo_actor/update/rs_filtered_fraction",
    ],
    "reward": ["ppo_actor/final_reward/avg"],
    "entropy": ["ppo_actor/update/entropy/avg"],
    "imp_w": ["ppo_actor/update/importance_weight/avg"],
    "kl": ["ppo_actor/update/compute_logp/kl_div_dual/avg"],
    "grad_norm": ["ppo_actor/update/grad_norm"],
    "acc_proxy": ["ppo_actor/correct_n_seqs"],
}


def parse_series(text: str) -> dict[str, list[float]]:
    """Parse box-drawing tables by splitting on │ (not overlapping finditer)."""
    text = ANSI.sub("", text)
    series: dict[str, list[float]] = defaultdict(list)
    for line in text.splitlines():
        if "│" not in line:
            continue
        parts = [p.strip() for p in line.split("│")]
        # parts[0] and parts[-1] are usually empty around outer borders
        i = 0
        while i + 1 < len(parts):
            k, v = parts[i], parts[i + 1]
            if k and NUM.match(v):
                series[k].append(float(v))
                i += 2
            else:
                i += 1
    return series


def pick(series: dict[str, list[float]], aliases: list[str]) -> list[float] | None:
    for a in aliases:
        if a in series and series[a]:
            return series[a]
    return None


def window_means(xs: list[float], win: int) -> list[tuple[str, float]]:
    out = []
    n = len(xs)
    if n == 0:
        return out
    for i in range(0, n, win):
        chunk = xs[i : i + win]
        label = f"{i}-{i + len(chunk) - 1}"
        out.append((label, sum(chunk) / len(chunk)))
    out.append(("all", sum(xs) / n))
    return out


def summarize(path: Path, win: int, batch: int) -> str:
    text = path.read_text(errors="replace")
    series = parse_series(text)
    lines = [f"# {path.name}", f"keys_found={len(series)}"]
    for name, aliases in KEYS.items():
        xs = pick(series, aliases)
        if xs is None:
            lines.append(f"{name}: MISSING (tried {aliases})")
            continue
        if name == "acc_proxy":
            acc = [v / batch for v in xs]
            lines.append(f"acc(~correct/{batch}): n={len(acc)}")
            for lab, m in window_means(acc, win):
                lines.append(f"  {lab}: {m:.4f}")
        else:
            lines.append(f"{name}: n={len(xs)} first={xs[0]:.5g} last={xs[-1]:.5g}")
            for lab, m in window_means(xs, win):
                lines.append(f"  {lab}: {m:.5g}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--win", type=int, default=40)
    ap.add_argument("--batch", type=int, default=72)
    args = ap.parse_args()
    for p in args.logs:
        print(summarize(p, args.win, args.batch))
        print()


if __name__ == "__main__":
    main()
