from __future__ import annotations

import pytest
import torch

areal = pytest.importorskip("areal")

from areal.api.cli_args import NormConfig, RejectionSamplingConfig  # noqa: E402
from areal.utils.data import Normalization  # noqa: E402
from areal.utils.functional import apply_rejection_sampling  # noqa: E402


@pytest.mark.integration
def test_running_mean_uses_previous_batches_as_baseline() -> None:
    norm = Normalization(
        NormConfig(mean_level="running", std_level=None, running_window=2)
    )
    first = norm(torch.tensor([1.0, 3.0]))
    second = norm(torch.tensor([5.0, 7.0]))
    assert torch.allclose(first, torch.tensor([-1.0, 1.0]))
    assert torch.allclose(second, torch.tensor([3.0, 5.0]))


@pytest.mark.integration
def test_dis_masks_out_of_band_tokens() -> None:
    loss_mask = torch.ones(1, 3, dtype=torch.bool)
    result = apply_rejection_sampling(
        proximal_logprobs=torch.log(torch.tensor([[0.5, 1.0, 7.0]])),
        old_logprobs=torch.zeros(1, 3),
        loss_mask=loss_mask,
        cu_seqlens=None,
        config=RejectionSamplingConfig(
            level="token",
            metric="ratio",
            action="mask",
            lower=0.3,
            upper=5.0,
        ),
    )
    assert result.loss_mask.tolist() == [[False, True, False]]
    assert result.filtered_fraction == pytest.approx(2 / 3)
