"""CPU unit tests for DIS ratio / mask (Phase 0 exit)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sao.dis import dis_mask, dis_token_weight, importance_ratio, mask_ratio


def test_ratio_identity():
    logp = torch.tensor([-0.2, -1.0, -0.5])
    r = importance_ratio(logp, logp)
    assert torch.allclose(r, torch.ones_like(r), atol=1e-6)


def test_dis_mask_bounds_math_defaults():
    ratio = torch.tensor([0.2, 0.3, 1.0, 5.0, 5.1])
    m = dis_mask(ratio, lower=0.3, upper=5.0)
    assert m.tolist() == [False, True, True, True, False]


def test_dis_weight_zeros_outside():
    ratio = torch.tensor([0.2, 1.2, 10.0])
    w = dis_token_weight(ratio, lower=0.3, upper=5.0)
    assert w[0].item() == 0.0
    assert abs(w[1].item() - 1.2) < 1e-6
    assert w[2].item() == 0.0


def test_mask_ratio_metric():
    ratio = torch.tensor([0.5, 1.0, 10.0, 1.0])
    m = dis_mask(ratio, 0.3, 5.0)
    # 1 of 4 outside → mask_out ratio 0.25
    assert abs(mask_ratio(m).item() - 0.25) < 1e-6


if __name__ == "__main__":
    test_ratio_identity()
    test_dis_mask_bounds_math_defaults()
    test_dis_weight_zeros_outside()
    test_mask_ratio_metric()
    print("OK: DIS CPU tests passed")
