"""Tests for DIS policy weight helper."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sao.loss import dis_policy_weights


def test_outside_range_zero_weight():
    log_theta = torch.tensor([0.0, 0.0, 0.0])
    # ratios: exp(0-0)=1, exp(0-(-2))~7.39 > 5, exp(0-2)~0.135 < 0.3
    log_roll = torch.tensor([0.0, -2.0, 2.0])
    adv = torch.tensor([1.0, 1.0, 1.0])
    w, m = dis_policy_weights(log_theta, log_roll, adv, lower=0.3, upper=5.0)
    assert abs(w[0].item() - 1.0) < 1e-5
    assert w[1].item() == 0.0
    assert w[2].item() == 0.0
    assert abs(m["dis/mask_ratio"].item() - (2 / 3)) < 1e-5


if __name__ == "__main__":
    test_outside_range_zero_weight()
    print("OK: DIS loss tests passed")
