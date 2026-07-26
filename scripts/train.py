#!/usr/bin/env python3
"""AReaL RLVR training entry shared by all MATH experiment settings."""

from __future__ import annotations

import os
import sys

from areal import PPOTrainer
from areal.api.cli_args import PPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, PPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )

    reward_config = config.reward
    reward_root = os.environ.get("SAO_REWARD_FS_ROOT")
    if (
        reward_config is not None
        and getattr(reward_config, "backend", "local") == "fs_shard"
        and reward_root
    ):
        reward_config.fs_shard.root = reward_root
        if run_id := os.environ.get("SAO_REWARD_RUN_ID"):
            reward_config.fs_shard.run_id = run_id

    workflow_kwargs = {
        "reward_fn": "areal.reward.gsm8k.gsm8k_reward_fn",
        "gconfig": config.gconfig,
        "tokenizer": config.tokenizer_path,
        "enable_thinking": False,
        "reward_config": reward_config,
    }
    with PPOTrainer(config, train_dataset=train_dataset, valid_dataset=None) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
