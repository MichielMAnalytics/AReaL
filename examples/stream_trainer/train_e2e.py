# SPDX-License-Identifier: Apache-2.0
"""Real SGLang/AWEX smoke test with an artificial delayed prompt group."""

import asyncio
import os
import sys
from typing import Any

from datasets import Dataset

from areal import PPOTrainer
from areal.api import InferenceEngine
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.workflow.rlvr import RLVRWorkflow


class DelayedWorkflow(RLVRWorkflow):
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any]:
        # Test-only delay precedes a real model request. Other groups can drain
        # while this server still owns a live session and the old policy.
        await asyncio.sleep(float(data["delay_seconds"]))
        return await super().arun_episode(engine, data)


def constant_reward(*args, **kwargs) -> float:
    return 1.0


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, GRPOConfig)
    delay = float(os.environ.get("STREAM_TEST_TAIL_SECONDS", "60"))
    if delay <= 0:
        raise ValueError("STREAM_TEST_TAIL_SECONDS must be positive")
    batch = config.train_dataset.batch_size
    if batch < 2:
        raise ValueError("The E2E smoke test needs at least two prompt groups")
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": f"Compute {index} + 1. Answer briefly."}
                ],
                "delay_seconds": delay if index == batch - 1 else 0.0,
            }
            for index in range(batch)
        ]
    )
    with PPOTrainer(config, train_dataset=dataset, valid_dataset=None) as trainer:
        trainer.train(
            workflow="examples.stream_trainer.train_e2e.DelayedWorkflow",
            workflow_kwargs={
                "reward_fn": "examples.stream_trainer.train_e2e.constant_reward",
                "gconfig": config.gconfig,
                "tokenizer": config.tokenizer_path,
                "enable_thinking": False,
            },
        )


if __name__ == "__main__":
    from examples.stream_trainer.train_e2e import main as canonical_main

    canonical_main(sys.argv[1:])
