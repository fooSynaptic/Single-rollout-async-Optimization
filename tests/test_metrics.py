from __future__ import annotations

from sao.metrics import parse_log


def test_parse_areal_metrics(tmp_path) -> None:
    log = tmp_path / "phase2_hard_sao.log"
    log.write_text(
        """
20260725-13:51:09.653 StatsLogger INFO: Train step 1/1000 done.
│ ppo_actor/correct_n_seqs   │  3.0000e+00 │ ppo_actor/incorrect_n_seqs │  1.0000e+00 │
│ rollout/reward             │  7.5000e-01 │ ppo_actor/update/dis/mask_ratio │ 2.5e-01 │
20260725-13:52:09.653 StatsLogger INFO: Train step 2/1000 done.
│ ppo_actor/correct_n_seqs   │  2.0000e+00 │ ppo_actor/incorrect_n_seqs │  2.0000e+00 │
│ rollout/reward             │  5.0000e-01 │ ppo_critic/update/value_loss │ 1.2e-01 │
Training completes!
"""
    )
    result = parse_log(log)
    assert result["last_step"] == 2
    assert result["series"]["acc"] == [
        {"step": 1, "value": 0.75},
        {"step": 2, "value": 0.5},
    ]
    assert result["series"]["reward"][-1]["value"] == 0.5
    assert result["health"]["completed"] is True
