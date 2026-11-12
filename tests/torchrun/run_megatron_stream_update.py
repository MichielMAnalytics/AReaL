# SPDX-License-Identifier: Apache-2.0
"""GPU-only comparison of real Megatron PPO updates; see examples/stream_trainer.

This uses the real distributed Adam optimizer. Public DP-zero gather/scatter
APIs preserve FP32 masters and Adam moments, which state_dict alone omits.
Only DP roots retain complete optimizer tensors. No optimizer method is patched.
"""

import argparse
import copy
import json
import math
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

_SUM_FIELDS = (
    "reference_squared_norm",
    "observed_squared_norm",
    "dot_product",
    "squared_difference",
    "elements",
    "strict_mismatched_elements",
    "nonfinite_elements",
)


def _derived_metrics(sums: dict[str, float | int]) -> dict[str, float | None]:
    x2 = sums["reference_squared_norm"]
    y2 = sums["observed_squared_norm"]
    xy = sums["dot_product"]
    difference = sums["squared_difference"]
    finite = all(math.isfinite(value) for value in (x2, y2, xy, difference))
    xnorm = math.sqrt(max(0.0, x2)) if finite else 0.0
    ynorm = math.sqrt(max(0.0, y2)) if finite else 0.0
    scale = xy / x2 if finite and x2 > 0 else None
    residual = max(0.0, y2 - xy * xy / x2) if scale is not None else None
    count = sums["elements"]
    return {
        "relative_l2": math.sqrt(max(0.0, difference)) / xnorm
        if finite and xnorm > 0
        else (0.0 if finite and difference == 0 else None),
        "cosine": max(-1.0, min(1.0, xy / (xnorm * ynorm)))
        if finite and xnorm > 0 and ynorm > 0
        else None,
        "norm_ratio_observed_to_reference": ynorm / xnorm
        if finite and xnorm > 0
        else None,
        "best_scale_reference_to_observed": scale,
        "best_scale_residual_relative_l2": math.sqrt(residual) / ynorm
        if residual is not None and ynorm > 0
        else None,
        "strict_mismatch_rate": sums["strict_mismatched_elements"] / count
        if count
        else 0.0,
    }


def _tensor_diagnostics(
    reference: torch.Tensor, observed: torch.Tensor, rtol: float, atol: float
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "observed_shape": list(observed.shape),
        "reference_dtype": str(reference.dtype),
        "observed_dtype": str(observed.dtype),
    }
    try:
        torch.testing.assert_close(reference, observed, rtol=rtol, atol=atol)
        report["strict_pass"] = True
    except AssertionError as error:
        report.update(strict_pass=False, strict_error=str(error)[:2000])
    if reference.shape != observed.shape:
        return report
    sums: dict[str, float | int] = {name: 0 for name in _SUM_FIELDS}
    maximum = 0.0
    xflat = reference.detach().cpu().reshape(-1)
    yflat = observed.detach().cpu().reshape(-1)
    # FP64 sufficient statistics can be summed across ranks. Bound transient CPU
    # memory rather than materializing several FP64 copies of a full model shard.
    for start in range(0, xflat.numel(), 1 << 20):
        x = xflat[start : start + (1 << 20)].double()
        y = yflat[start : start + (1 << 20)].double()
        difference = y - x
        sums["reference_squared_norm"] += x.square().sum().item()
        sums["observed_squared_norm"] += y.square().sum().item()
        sums["dot_product"] += (x * y).sum().item()
        sums["squared_difference"] += difference.square().sum().item()
        sums["elements"] += x.numel()
        sums["strict_mismatched_elements"] += (
            (~torch.isclose(x, y, rtol=rtol, atol=atol, equal_nan=False)).sum().item()
        )
        sums["nonfinite_elements"] += (
            (~(torch.isfinite(x) & torch.isfinite(y))).sum().item()
        )
        maximum = max(maximum, difference.abs().max().item())
    report.update(sums=sums, max_abs=maximum, **_derived_metrics(sums))
    return report


def comparison_diagnostics(
    reference: Any, observed: Any, rtol: float, atol: float
) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    errors: list[str] = []

    def compare(left: Any, right: Any, path: str) -> None:
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            tensors[path] = _tensor_diagnostics(left, right, rtol, atol)
        elif isinstance(left, Mapping) and isinstance(right, Mapping):
            if left.keys() != right.keys():
                errors.append(f"{path}: mapping keys differ")
            for key in left:
                if key in right:
                    compare(left[key], right[key], f"{path}/{key}")
        elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                errors.append(f"{path}: sequence lengths differ")
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f"{path}/{index}")
        elif type(left) is not type(right) or left != right:
            errors.append(f"{path}: scalar value or type differs")

    compare(reference, observed, "")
    sums = {name: 0 for name in _SUM_FIELDS}
    for tensor in tensors.values():
        for name, value in tensor.get("sums", {}).items():
            sums[name] += value
    return {
        "strict_pass": not errors
        and all(tensor["strict_pass"] for tensor in tensors.values()),
        "structural_errors": errors,
        "tensors": tensors,
        "aggregate": {
            "sums": sums,
            "max_abs": max(
                (tensor.get("max_abs", 0.0) for tensor in tensors.values()), default=0.0
            ),
            **_derived_metrics(sums),
        },
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    return value


def write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


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

    def group(
        self, group_id: int, *, power_of_two_mass: bool = False
    ) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(1234 + group_id)
        length = 32 + group_id * 8
        tokens = torch.randint(100, 1000, (2, length), generator=generator)
        loss_mask = torch.ones_like(tokens)
        prefix = (16, 16, 24)[group_id] if power_of_two_mass else 8
        loss_mask[:, :prefix] = 0
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
    parser.add_argument(
        "--power-of-two-mass",
        action="store_true",
        help="DP2-only normalization-rounding control: mass128, local masses80/48",
    )
    args = parser.parse_args()
    world = int(os.environ["WORLD_SIZE"])
    if args.tp_size < 1 or world % args.tp_size or world // args.tp_size < 2:
        parser.error("Require at least two DP replicas and WORLD_SIZE divisible by TP")
    dp = world // args.tp_size
    if args.power_of_two_mass and dp != 2:
        parser.error("--power-of-two-mass requires exactly two DP replicas")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
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
        # Lengths 32/40/48 each occupy one real microbatch in both paths.
        # Isolate accumulation from BF16 GEMM/GDN repacking differences.
        mb_spec=MicroBatchSpec(max_tokens_per_mb=48),
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
    replica_topology = actor.stream_topology()
    expected_replica_ranks = set(
        dist.get_process_group_ranks(actor.cpu_model_parallel_group)
    )
    assert isinstance(replica_topology, list), (
        "Topology must gather all replica identities"
    )
    assert {row["rank"] for row in replica_topology} == expected_replica_ranks
    assert len(replica_topology) == len(expected_replica_ranks)
    assert all(row["gpu_uuid"] and row["host"] for row in replica_topology)
    rank = actor.data_parallel_rank
    raw = [actor.group(rank * 2, power_of_two_mass=args.power_of_two_mass)]
    if rank == 0:
        raw.append(actor.group(rank * 2 + 1, power_of_two_mass=args.power_of_two_mass))
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
    local_token_mass = 0
    for chunk_id, group in enumerate(raw):
        chunk_summary = actor.stream_accumulate(
            "gpu-parity", rank * 2 + chunk_id, [copy.deepcopy(group)]
        )
        local_token_mass += chunk_summary["token_mass"]
    backward_done = time.monotonic() - started
    # This barrier is test-only: production coordinator waits for rollout completion.
    dist.barrier()
    finish_summary = actor.finish_stream_update("gpu-parity")
    global_token_mass = int(finish_summary["stream_token_mass"])
    fixture = (
        "fixed-shape-power-of-two-mass"
        if args.power_of_two_mass
        else "fixed-shape-default-mass"
    )
    diagnostics_path = output / f"diagnostics-rank-{dist.get_rank()}.json"
    diagnostics: dict[str, Any] = {
        "rank": dist.get_rank(),
        "dp_rank": rank,
        "fixture": fixture,
        "actual_local_token_mass": local_token_mass,
        "actual_global_token_mass": global_token_mass,
        "rtol": args.rtol,
        "atol": args.atol,
        "status": "collecting",
        "comparison_direction": "reference=serial; observed=stream",
        "sum_scope": "sum gradient aggregate.sums across ranks; TP-replicated optimizer entries remain counted per rank",
        "replica_topology": replica_topology,
        "comparisons": {},
    }
    # Diagnose and persist gradients before any final parity assertion. A gradient
    # mismatch must not prevent inspecting the actual model/Adam update outcome.
    diagnostics["comparisons"]["gradients"] = comparison_diagnostics(
        gradients, actor.observed_gradients, args.rtol, args.atol
    )
    write_report(diagnostics_path, diagnostics)
    observed_state = actor.snapshot()  # All ranks still participate in DP-zero gather.
    for category in ("model", "optimizer", "optimizer_parameters", "scheduler"):
        diagnostics["comparisons"][category] = comparison_diagnostics(
            reference[category], observed_state[category], args.rtol, args.atol
        )
        write_report(diagnostics_path, diagnostics)
    overlap_pass = rank == dp - 1 or backward_done < args.delay_seconds
    mass_pass = not args.power_of_two_mass or (
        global_token_mass == 128 and local_token_mass == (80 if rank == 0 else 48)
    )
    diagnostics["overlap_pass"] = overlap_pass
    diagnostics["fixture_mass_pass"] = mass_pass
    diagnostics["backward_done_s"] = backward_done
    local_failed = not (
        overlap_pass
        and mass_pass
        and all(report["strict_pass"] for report in diagnostics["comparisons"].values())
    )
    diagnostics["local_status"] = "failed" if local_failed else "passed"
    # The global decision is delayed until every rank has written diagnostics;
    # no rank is advertised as passed when another failed a strict comparison.
    failed_ranks = torch.tensor(
        int(local_failed), dtype=torch.int32, device=actor.device
    )
    dist.all_reduce(failed_ranks, op=dist.ReduceOp.SUM)
    global_failures = int(failed_ranks.item())
    diagnostics["global_failed_ranks"] = global_failures
    diagnostics["status"] = "failed" if global_failures else "passed"
    write_report(diagnostics_path, diagnostics)
    torch.cuda.synchronize()
    result = {
        "rank": dist.get_rank(),
        "dp_rank": rank,
        "status": diagnostics["status"],
        "fixture": fixture,
        "actual_local_token_mass": local_token_mass,
        "actual_global_token_mass": global_token_mass,
        "backward_done_s": backward_done,
        "rtol": args.rtol,
        "atol": args.atol,
        "optimizer": "distributed Adam; DP-zero FP32 masters and nonempty moments",
        "optimizer_parameter_state_owner": "DP zero of each optimizer/TP shard",
        "diagnostics": diagnostics_path.name,
        "comparison_pass": {
            category: report["strict_pass"]
            for category, report in diagnostics["comparisons"].items()
        },
        "global_failed_ranks": global_failures,
    }
    write_report(output / f"rank-{dist.get_rank()}.json", result)
    dist.barrier()
    actor.destroy()
    if global_failures:
        raise AssertionError(
            f"Strict Stream Trainer parity failed on {global_failures} ranks; "
            f"see {diagnostics_path} (gradient and updated-state checks retained)"
        )


if __name__ == "__main__":
    main()
