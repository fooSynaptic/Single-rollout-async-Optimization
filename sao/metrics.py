"""Parse AReaL training logs into experiment curves and health signals."""

from __future__ import annotations

import re
from pathlib import Path

STEP_RE = re.compile(r"Train step (\d+)/(\d+) done")
NUMBER = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
METRICS = {
    "reward": ("rollout/reward", "ppo_actor/task_reward/avg"),
    "actor_loss": ("ppo_actor/update/actor_loss/avg", "ppo_actor/update/pg_loss"),
    "critic_loss": ("ppo_critic/update/critic_loss/avg", "ppo_critic/update/value_loss"),
    "dis_mask": ("ppo_actor/update/dis/mask_ratio", "ppo_actor/update/rs_filtered_fraction"),
    "correct": ("ppo_actor/correct_n_seqs",),
    "incorrect": ("ppo_actor/incorrect_n_seqs",),
}
EXPERIMENT_PATTERNS = {
    "sao": "phase2_hard_sao",
    "grpo": "phase2_hard_grpo",
    "grpo_dis": "phase2_hard_grpo_dis",
    "grpo_dis_g1": "phase2_hard_grpo_dis_g1",
    "running_mean": "phase2_hard_running_mean",
}


def _metric(block: str, aliases: tuple[str, ...]) -> float | None:
    for alias in aliases:
        match = re.search(re.escape(alias) + r"\s*│\s*" + NUMBER, block)
        if match:
            return float(match.group(1))
    return None


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    matches = list(STEP_RE.finditer(text))
    series = {key: [] for key in (*METRICS, "acc")}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        # AReaL's stats table follows the "Train step ... done" marker.
        block = text[match.end():end]
        step = int(match.group(1))
        values = {key: _metric(block, aliases) for key, aliases in METRICS.items()}
        for key, value in values.items():
            if value is not None:
                series[key].append({"step": step, "value": value})
        correct, incorrect = values["correct"], values["incorrect"]
        if correct is not None and incorrect is not None and correct + incorrect > 0:
            series["acc"].append(
                {"step": step, "value": correct / (correct + incorrect)}
            )

    tail = text[-200_000:]
    return {
        "path": str(path),
        "last_step": int(matches[-1].group(1)) if matches else 0,
        "total_steps": int(matches[-1].group(2)) if matches else 0,
        "series": series,
        "health": {
            "oom": bool(re.search(r"CUDA out of memory|OutOfMemory|low on memory", tail, re.I)),
            "nan_or_inf": bool(re.search(r"\b(?:nan|inf)\b", tail, re.I)),
            "completed": "Training completes!" in tail,
        },
    }


def latest_logs(log_dir: Path) -> dict[str, Path]:
    result = {}
    files = sorted(log_dir.glob("*.log"), key=lambda path: path.stat().st_mtime)
    for name, marker in EXPERIMENT_PATTERNS.items():
        candidates = [path for path in files if marker in path.name]
        if name == "grpo":
            candidates = [path for path in candidates if "grpo_dis" not in path.name]
        elif name == "grpo_dis":
            candidates = [path for path in candidates if "grpo_dis_g1" not in path.name]
        if candidates:
            result[name] = candidates[-1]
    return result


def collect_metrics(log_dir: Path) -> dict:
    experiments = {name: parse_log(path) for name, path in latest_logs(log_dir).items()}
    return {"log_dir": str(log_dir), "experiments": experiments}
