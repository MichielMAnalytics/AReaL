# SPDX-License-Identifier: Apache-2.0
"""GPU-only comparison of real Megatron PPO updates; see examples/stream_trainer.

This intentionally uses the non-distributed Adam optimizer so its public
state_dict includes moments and FP32 master parameters. The E2E recipe also
exercises the normal distributed optimizer. No optimizer method is patched.
"""

import argparse
import copy
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    DistributedDataParallelConfig,
    MegatronEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    PPOActorConfig,
)
from areal.engine.megatron_engine import MegatronPPOActor
from areal.utils.data import tensor_container_to


class UpdateProbe(MegatronPPOActor):
    """Observe real post-reduction, pre-clipping gradients at the step boundary."""

    def optimizer_step(self) -> dict[str, float]:
        self.observed_gradients = {
            f"{index}.{name}": parameter.main_grad.detach().cpu().clone()
            for index, chunk in enumerate(self.model)
            for name, parameter in chunk.named_parameters()
            if parameter.requires_grad and hasattr(parameter, "main_grad")
        }
        return super().optimizer_step()

    def snapshot(self) -> dict[str, Any]:
        # Stage before cloning: duplicating Adam/model tensors on the GPU
        # would make a small validation model require nearly twice the memory.
        return copy.deepcopy(
            tensor_container_to(
                {
                    "model": [chunk.state_dict() for chunk in self.model],
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.lr_scheduler.state_dict(),
                },
                "cpu",
            ),
        )

    def tensor_items(self, value: Any, path: str = ""):
        if isinstance(value, torch.Tensor):
            yield path, value
        elif isinstance(value, Mapping):
            for key, child in value.items():
                yield from self.tensor_items(child, f"{path}/{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                yield from self.tensor_items(child, f"{path}/{index}")

    def restore(self, state: dict[str, Any]) -> None:
        for chunk, chunk_state in zip(self.model, state["model"], strict=True):
            chunk.load_state_dict(chunk_state)
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        self.lr_scheduler.load_state_dict(copy.deepcopy(state["scheduler"]))
        self.optimizer_zero_grad()

    def serial_update(self, raw: list[dict[str, Any]]) -> None:
        data = copy.deepcopy(raw)
        logps = self.compute_logp(data)
        if logps is None:
            raise AssertionError("PP=1 must return logprobs")
        for trajectory, logp in zip(data, logps, strict=True):
            trajectory["prox_logp"] = logp
        self.ppo_update(self.compute_advantages(data))

    def assert_same(self, left: Any, right: Any, rtol: float, atol: float) -> None:
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
        elif isinstance(left, Mapping):
            assert left.keys() == right.keys()
            for key in left:
                self.assert_same(left[key], right[key], rtol, atol)
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right, strict=True):
                self.assert_same(a, b, rtol, atol)
        else:
            assert left == right, (left, right)

    def group(self, group_id: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(1234 + group_id)
        length = 32 + group_id * 8
        tokens = torch.randint(100, 1000, (2, length), generator=generator)
        loss_mask = torch.ones_like(tokens)
        loss_mask[:, :8] = 0
        return {
            "input_ids": tokens,
            "attention_mask": torch.ones_like(tokens, dtype=torch.bool),
            "loss_mask": loss_mask,
            "logprobs": torch.zeros_like(tokens, dtype=torch.float32),
            "rewards": torch.tensor([1.0, 0.5]),
            "is_truncated": torch.zeros(2, dtype=torch.bool),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--bridge-type", default="mbridge")
    parser.add_argument("--delay-seconds", type=float, default=10)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    world = int(os.environ["WORLD_SIZE"])
    if args.tp_size < 1 or world % args.tp_size or world // args.tp_size < 2:
        parser.error("Require at least two DP replicas and WORLD_SIZE divisible by TP")
    dp = world // args.tp_size
    backend = f"megatron:d{dp}t{args.tp_size}p1"
    torch.manual_seed(1234)
    config = PPOActorConfig(
        backend=backend,
        experiment_name="stream-update-check",
        trial_name="gpu",
        path=args.model_path,
        dtype="bfloat16",
        disable_dropout=True,
        gradient_checkpointing=True,
        ppo_n_minibatches=1,
        kl_ctl=0,
        reward_norm=None,
        adv_norm=None,
        use_decoupled_loss=False,
        recompute_logprob=True,
        weight_update_mode="awex",
        mb_spec=MicroBatchSpec(max_tokens_per_mb=128),
        optimizer=OptimizerConfig(lr=1e-4, lr_scheduler_type="constant"),
        megatron=MegatronEngineConfig(
            bridge_type=args.bridge_type,
            ddp=DistributedDataParallelConfig(
                use_distributed_optimizer=False,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
            ),
        ),
    )
    actor = UpdateProbe(config)
    actor.create_process_group(ModelAllocation.from_str(backend).parallel)
    actor.initialize(
        None,
        FinetuneSpec(total_train_epochs=1, dataset_size=32, train_batch_size=2 * dp),
    )
    actor._ensure_awex_adapter()
    rank = actor.data_parallel_rank
    raw = [actor.group(rank * 2)]
    if rank == 0:
        raw.append(actor.group(rank * 2 + 1))
    # Match sampler logprobs to this policy to avoid an all-clipped, zero-gradient check.
    for group, logp in zip(raw, actor.compute_logp(copy.deepcopy(raw)), strict=True):
        group["logprobs"] = torch.roll(logp.detach().cpu(), shifts=1, dims=-1)
    actor.serial_update(raw)  # Populate Adam moments before either compared update.
    initial = actor.snapshot()
    moments = [
        value
        for name, value in actor.tensor_items(initial["optimizer"])
        if name.endswith("/exp_avg")
    ]
    assert moments and any(torch.count_nonzero(value) for value in moments), (
        "Optimizer snapshot must contain nonzero Adam moments, not just param groups"
    )
    actor.serial_update(raw)
    reference = actor.snapshot()
    gradients = copy.deepcopy(actor.observed_gradients)
    assert gradients and any(torch.count_nonzero(value) for value in gradients.values())
    actor.restore(initial)
    actor.assert_same(initial, actor.snapshot(), 0, 0)
    dist.barrier()
    started = time.monotonic()
    if rank == dp - 1:
        time.sleep(args.delay_seconds)
    for chunk_id, group in enumerate(raw):
        actor.stream_accumulate(
            "gpu-parity", rank * 2 + chunk_id, [copy.deepcopy(group)]
        )
    backward_done = time.monotonic() - started
    # This barrier is test-only: production coordinator waits for rollout completion.
    dist.barrier()
    actor.finish_stream_update("gpu-parity")
    actor.assert_same(gradients, actor.observed_gradients, args.rtol, args.atol)
    actor.assert_same(reference, actor.snapshot(), args.rtol, args.atol)
    if rank < dp - 1:
        assert backward_done < args.delay_seconds, (
            "Increase delay to observe overlap on this GPU/model"
        )
    torch.cuda.synchronize()
    result = {
        "rank": dist.get_rank(),
        "dp_rank": rank,
        "status": "passed",
        "backward_done_s": backward_done,
        "rtol": args.rtol,
        "atol": args.atol,
        "optimizer": "non-distributed Adam; nonempty moments",
        "checks": [
            "same initial model/optimizer/scheduler",
            "pre-step gradients",
            "updated model",
            "optimizer state",
            "scheduler state",
        ],
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank-{dist.get_rank()}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    dist.barrier()
    actor.destroy()


if __name__ == "__main__":
    main()
