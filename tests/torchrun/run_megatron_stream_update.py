# SPDX-License-Identifier: Apache-2.0
"""GPU-only comparison of real Megatron PPO updates; see examples/stream_trainer.

This uses the real distributed Adam optimizer. Public DP-zero gather/scatter
APIs preserve FP32 masters and Adam moments, which state_dict alone omits.
Only DP roots retain complete optimizer tensors. No optimizer method is patched.
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
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

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
from areal.engine.megatron_utils.optimizer_chain import iter_megatron_optimizer_leaves
from areal.utils.data import tensor_container_to


class UpdateProbe(MegatronPPOActor):
    """Observe real post-reduction, pre-clipping gradients at the step boundary."""

    def distributed_optimizers(self) -> list[DistributedOptimizer]:
        optimizers = list(iter_megatron_optimizer_leaves(self.optimizer))
        if not optimizers or not all(
            isinstance(optimizer, DistributedOptimizer) for optimizer in optimizers
        ):
            raise AssertionError(
                "Numerical parity requires real distributed optimizer leaves"
            )
        return optimizers

    def optimizer_step(self) -> dict[str, float]:
        optimizers = self.distributed_optimizers()
        if any(optimizer.grad_scaler is not None for optimizer in optimizers):
            raise AssertionError("This BF16 probe requires no dynamic gradient scaler")
        # After reduce-scatter, non-owned regions of model.main_grad are not
        # global gradients. Public prepare_grads exposes only the valid owned
        # shards on optimizer parameters. With BF16/no scaler this is an
        # idempotent copy/view operation: the ordinary step prepares them again.
        if self.optimizer.prepare_grads():
            raise AssertionError("Unexpected nonfinite gradients before comparison")
        self.observed_gradients = {}
        for optimizer_index, optimizer in enumerate(optimizers):
            for parameter_index, parameter in enumerate(optimizer.get_parameters()):
                gradient = getattr(parameter, "decoupled_grad", None)
                if gradient is None:
                    gradient = parameter.grad
                if gradient is not None:
                    self.observed_gradients[f"{optimizer_index}.{parameter_index}"] = (
                        gradient.detach().cpu().clone()
                    )
        return super().optimizer_step()

    def snapshot(self) -> dict[str, Any]:
        # Every DP rank participates, but complete master/moment buffers stay
        # only on DP zero for each TP shard. Do not broadcast or deepcopy these
        # already-independent CPU gather buffers onto the other DP ranks.
        parameter_states = [
            optimizer.get_parameter_state_dp_zero()
            for optimizer in self.distributed_optimizers()
        ]
        return {
            "model": copy.deepcopy(
                tensor_container_to([chunk.state_dict() for chunk in self.model], "cpu")
            ),
            "optimizer": copy.deepcopy(
                tensor_container_to(self.optimizer.state_dict(), "cpu")
            ),
            "optimizer_parameters": parameter_states,
            "scheduler": copy.deepcopy(self.lr_scheduler.state_dict()),
        }

    def assert_parameter_state_complete(self, state: dict[str, Any]) -> None:
        for optimizer, parameter_state in zip(
            self.distributed_optimizers(), state["optimizer_parameters"], strict=True
        ):
            is_root = dist.get_rank(optimizer.data_parallel_group_gloo) == 0
            if not is_root:
                assert parameter_state is None, (
                    "Full optimizer tensors must remain on DP zero"
                )
                continue
            assert parameter_state is not None, (
                "DP zero must own complete optimizer state"
            )
            for field in ("param", "exp_avg", "exp_avg_sq"):
                tensors = [
                    value
                    for name, value in self.tensor_items(parameter_state)
                    if name.endswith(f"/{field}")
                ]
                assert tensors and any(
                    torch.count_nonzero(value) for value in tensors
                ), f"DP-zero snapshot must contain nonzero {field} tensors"
                assert all(value.dtype == torch.float32 for value in tensors), (
                    f"DP-zero {field} tensors must preserve FP32 optimizer precision"
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
        for optimizer, parameter_state in zip(
            self.distributed_optimizers(), state["optimizer_parameters"], strict=True
        ):
            # All ranks call scatter; only DP zero supplies the full tensors.
            # FP8 format conversion can mutate the input checkpoint dictionary;
            # this BF16-only test intentionally excludes that conversion.
            if optimizer.ddp_config.fp8_param_gather:
                raise AssertionError(
                    "Numerical parity fixture requires BF16 parameter gather"
                )
            optimizer.load_parameter_state_from_dp_zero(parameter_state)
        self.lr_scheduler.load_state_dict(copy.deepcopy(state["scheduler"]))
        self.optimizer_zero_grad()

    def serial_update(self, raw: list[dict[str, Any]]) -> None:
        data = copy.deepcopy(raw)
        logps = self.compute_logp(data)
        if logps is None:
            raise AssertionError("PP=1 must return logprobs")
        for trajectory, logp in zip(data, logps, strict=True):
            trajectory["prox_logp"] = logp.detach().cpu()
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
                use_distributed_optimizer=True,
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
    actor.assert_parameter_state_complete(initial)
    actor.serial_update(raw)
    reference = actor.snapshot()
    actor.assert_parameter_state_complete(reference)
    for initial_parameters, reference_parameters in zip(
        initial["optimizer_parameters"], reference["optimizer_parameters"], strict=True
    ):
        if initial_parameters is not None:
            before = dict(actor.tensor_items(initial_parameters))
            after = dict(actor.tensor_items(reference_parameters))
            assert any(
                not torch.equal(value, after[name])
                for name, value in before.items()
                if name.endswith("/param")
            ), "Reference update must change FP32 master parameters"
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
        "optimizer": "distributed Adam; DP-zero FP32 masters and nonempty moments",
        "optimizer_parameter_state_owner": "DP zero of each optimizer/TP shard",
        "checks": [
            "same initial model/optimizer/scheduler",
            "owned pre-clipping gradient shards",
            "updated model",
            "optimizer metadata on all ranks",
            "DP-zero FP32 master parameters and Adam moments",
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
