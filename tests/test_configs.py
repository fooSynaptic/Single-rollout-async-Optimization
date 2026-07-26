from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text())


def test_sao_identity() -> None:
    config = load("phase2_hard_sao.yaml")
    assert config["gconfig"]["n_samples"] == 1
    assert config["critic"] is not None
    assert config["actor"]["rejection_sampling"]["metric"] == "ratio"


def test_grpo_identity() -> None:
    config = load("phase2_hard_grpo.yaml")
    assert config["gconfig"]["n_samples"] == 8
    assert "critic" not in config
    assert config["actor"]["rejection_sampling"] is None
    assert config["actor"]["reward_norm"]["mean_level"] == "group"


def test_grpo_dis_identity() -> None:
    config = load("phase2_hard_grpo_dis.yaml")
    assert config["gconfig"]["n_samples"] == 8
    assert config["actor"]["rejection_sampling"]["action"] == "mask"
    assert config["actor"]["path"].startswith("${MODEL_ROOT}")


def test_grpo_dis_g1_identity() -> None:
    config = load("phase2_hard_grpo_dis_g1.yaml")
    assert config["gconfig"]["n_samples"] == 1
    assert config["actor"]["rejection_sampling"]["action"] == "mask"
    assert config["actor"]["reward_norm"]["mean_level"] == "batch"


def test_running_mean_identity() -> None:
    config = load("phase2_hard_running_mean.yaml")
    assert config["gconfig"]["n_samples"] == 1
    assert config["actor"]["rejection_sampling"] is None
    assert config["actor"]["reward_norm"] == {
        "mean_level": "running",
        "std_level": None,
        "running_window": 128,
    }
