# SPDX-License-Identifier: Apache-2.0

"""Fixed-batch rollout/backward overlap with a single global commit."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from areal.utils import logging, stats_tracker

if TYPE_CHECKING:
    from areal.api.cli_args import StreamTrainerConfig
    from areal.infra.controller.rollout_controller import RolloutController
    from areal.infra.controller.train_controller import TrainController

logger = logging.getLogger("StreamTrainer")


class StreamTrainer:
    """Move only drained, physically matched replicas from inference to backward.

    The ordinary collector still owns batch membership and filtering. This
    coordinator sees only groups it actually consumes, and verifies that exact
    membership again before allowing an optimizer step. It never changes the
    training world or publishes partial weights.
    """

    def __init__(
        self,
        actor: TrainController,
        rollout: RolloutController,
        config: StreamTrainerConfig,
    ) -> None:
        self.actor = actor
        self.rollout = rollout
        self.config = config
        self._replicas: dict[int, tuple[int, list[int]]] = {}
        self._update_id: str | None = None
        self._started_at = 0.0
        self._collected = False

    @staticmethod
    def _gpu_key(host: str, gpu_uuid: str) -> tuple[str, str]:
        normalized = gpu_uuid.lower().removeprefix("gpu-")
        if not normalized:
            raise RuntimeError("Stream Trainer requires nonempty physical GPU UUIDs")
        return host, normalized

    def _discover_replicas(self) -> None:
        actors = self.actor.stream_topology()
        servers = self.rollout.stream_topology()
        actor_groups: dict[int, list[dict[str, Any]]] = {}
        for rank in actors:
            actor_groups.setdefault(rank["dp_rank"], []).append(rank)
        groups_by_gpus: dict[frozenset[tuple[str, str]], tuple[int, list[int]]] = {}
        all_actor_gpus: set[tuple[str, str]] = set()
        for dp_rank, members in actor_groups.items():
            ranks = sorted(member["rank"] for member in members)
            if any(sorted(member["replica_ranks"]) != ranks for member in members):
                raise RuntimeError(
                    "Actor ranks disagree about their complete TP replica"
                )
            if sum(self.actor.workers_is_dp_head[rank] for rank in ranks) != 1:
                raise RuntimeError("A streamed actor replica must have one DP head")
            gpus = frozenset(
                self._gpu_key(member["host"], member["gpu_uuid"]) for member in members
            )
            if len(gpus) != len(ranks) or all_actor_gpus.intersection(gpus):
                raise RuntimeError("Actor replicas overlap physical GPUs")
            all_actor_gpus.update(gpus)
            groups_by_gpus[gpus] = dp_rank, ranks
        replicas = {}
        for server in servers:
            worker_rank = server["worker_rank"]
            host = self.rollout.workers[worker_rank].ip
            gpus = frozenset(
                self._gpu_key(host, value) for value in server["gpu_uuids"]
            )
            if gpus not in groups_by_gpus:
                raise RuntimeError(
                    "SGLang GPU set does not exactly match an actor TP replica"
                )
            if worker_rank in replicas:
                raise RuntimeError("Duplicate SGLang worker in streamed topology")
            replicas[worker_rank] = groups_by_gpus.pop(gpus)
        if groups_by_gpus or len(replicas) != len(actor_groups):
            raise RuntimeError(
                "Stream Trainer requires one SGLang server per actor replica"
            )
        self._replicas = replicas

    def _event(self, name: str, **fields: Any) -> None:
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.info(
            "[StreamTrainer] event=%s update_id=%s elapsed_s=%.6f %s",
            name,
            self._update_id,
            time.monotonic() - self._started_at,
            details,
        )

    def _backward(self, event: dict[str, Any], worker_rank: int) -> float:
        dp_rank, ranks = self._replicas[worker_rank]
        chunk_id = event["task_id"]
        self._event("backward_start", chunk_id=chunk_id, dp_rank=dp_rank)
        self.actor.stream_accumulate(
            self._update_id,
            chunk_id,
            [event["trajectory"]],
            worker_ranks=ranks,
        )
        completed_at = time.monotonic()
        self._event("backward_done", chunk_id=chunk_id, dp_rank=dp_rank)
        return completed_at

    def collect_and_accumulate(
        self,
        prepare_batch: Callable[[], list[dict[str, Any]]],
        *,
        global_step: int,
    ) -> list[dict[str, Any]]:
        if self._update_id is not None:
            raise RuntimeError("Commit or abort the previous streamed batch first")
        if not self._replicas:
            self._discover_replicas()
        self._update_id = f"step-{global_step}-v{self.rollout.get_version()}"
        self._started_at = time.monotonic()
        self._collected = False
        pending: deque[dict[str, Any]] = deque()
        observed: dict[int, int] = {}
        eligible: set[int] = set()
        drained: set[int] = set()
        jobs: dict[int, Future[float]] = {}
        completed: list[tuple[float, int]] = []
        collection_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="stream-collect"
        )
        backward_pool = ThreadPoolExecutor(
            max_workers=len(self._replicas), thread_name_prefix="stream-backward"
        )
        collection_started = False

        def collect() -> tuple[list[dict[str, Any]], float]:
            batch = prepare_batch()
            finished_at = time.monotonic()
            self._event("rollout_complete", groups=len(batch))
            return batch, finished_at

        def consume_events(timeout: float) -> None:
            for event in self.rollout.get_stream_completed(timeout=timeout):
                chunk_id = event["task_id"]
                if chunk_id in observed:
                    raise RuntimeError("A rollout group was consumed more than once")
                observed[chunk_id] = id(event["trajectory"])
                eligible.add(event["worker_rank"])
                pending.append(event)

        def reap() -> None:
            for rank, job in list(jobs.items()):
                if job.done():
                    completed.append((job.result(), rank))
                    del jobs[rank]

        def schedule() -> None:
            for rank in sorted(drained):
                if pending and rank not in jobs:
                    jobs[rank] = backward_pool.submit(
                        self._backward, pending.popleft(), rank
                    )

        try:
            self.rollout.begin_stream_collection()
            collection_started = True
            future = collection_pool.submit(collect)
            while not future.done():
                consume_events(self.config.poll_interval)
                reap()
                if pending:
                    # Only consider replicas that have completed a group; do
                    # not steal an unused replica before initial submission.
                    for rank in sorted(eligible - drained):
                        if self.rollout.try_drain_stream_replica(rank, keep_one=True):
                            drained.add(rank)
                            self._event(
                                "replica_drained",
                                worker_rank=rank,
                                dp_rank=self._replicas[rank][0],
                            )
                schedule()

            batch, rollout_finished_at = future.result()
            consume_events(0.0)
            if len(observed) != len(batch) or set(observed.values()) != {
                id(group) for group in batch
            }:
                raise RuntimeError(
                    "Streamed groups do not exactly match the collected training batch"
                )
            # No running workflow may remain at this fixed-batch boundary.
            # This also releases the last server exactly once, before any
            # actor operation on its physical GPUs.
            for rank in self.rollout.finish_stream_draining():
                if rank not in drained:
                    self._event(
                        "replica_drained",
                        worker_rank=rank,
                        dp_rank=self._replicas[rank][0],
                    )
                drained.add(rank)
            self.rollout.pause()
            while pending or jobs:
                reap()
                schedule()
                if jobs:
                    # Wait on one bounded RPC future at a time; RPC timeouts
                    # propagate and forbid a partial optimizer commit.
                    next(iter(jobs.values())).result()
            early = [
                (when, rank) for when, rank in completed if when < rollout_finished_at
            ]
            stats_tracker.scalar(
                **{
                    "stream_trainer/total_groups": len(batch),
                    "stream_trainer/early_groups": len(early),
                    "stream_trainer/early_replicas": len({rank for _, rank in early}),
                    "stream_trainer/rollout_seconds": rollout_finished_at
                    - self._started_at,
                    "stream_trainer/collection_and_backward_seconds": time.monotonic()
                    - self._started_at,
                }
            )
            self._collected = True
            return batch
        except BaseException:
            # A backward can fail after modifying gradients. Never fall back to
            # ordinary ppo_update on those buffers, or reopen inference memory.
            self.rollout.pause()
            for job in jobs.values():
                job.cancel()
            backward_pool.shutdown(wait=True, cancel_futures=True)
            try:
                self.actor.abort_stream_update(self._update_id)
            except Exception:
                logger.exception("Failed to discard an incomplete streamed update")
            raise
        finally:
            collection_pool.shutdown(wait=False, cancel_futures=True)
            backward_pool.shutdown(wait=True, cancel_futures=True)
            if collection_started:
                self.rollout.end_stream_collection()

    def commit(self) -> None:
        if self._update_id is None or not self._collected:
            raise RuntimeError("A complete streamed batch is required before commit")
        self._event("optimizer_start")
        self.actor.finish_stream_update(self._update_id)
        self._event("optimizer_done")
        self._update_id = None
        self._collected = False

    def release_rollout(self) -> None:
        """Clear admission fences after the ordinary global AWEX handoff."""
        if self._update_id is not None:
            raise RuntimeError(
                "Cannot release inference before a streamed update commits"
            )
        self.rollout.restore_stream_replicas()
