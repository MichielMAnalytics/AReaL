# SPDX-License-Identifier: Apache-2.0

"""Test streamed execution ordering with a collector gated on real backward."""

import queue
import threading
from types import SimpleNamespace

import pytest

from areal.api.cli_args import StreamTrainerConfig
from areal.trainer.stream_trainer import StreamTrainer


class _Actor:
    def __init__(self, collection_returned, *, fail_backward=False):
        self.workers_is_dp_head = [True, True]
        self.collection_returned = collection_returned
        self.first_backward = threading.Event()
        self.fail_backward = fail_backward
        self.accumulated = []
        self.commits = []
        self.aborts = []
        self.lock = threading.Lock()

    def stream_topology(self):
        # Physical placement deliberately reverses the rollout rank order.
        return [
            {
                "rank": 0,
                "dp_rank": 0,
                "replica_ranks": [0],
                "host": "node",
                "gpu_uuid": "GPU-slow",
            },
            {
                "rank": 1,
                "dp_rank": 1,
                "replica_ranks": [1],
                "host": "node",
                "gpu_uuid": "GPU-fast",
            },
        ]

    def stream_accumulate(self, update_id, chunk_id, data, *, worker_ranks):
        with self.lock:
            self.accumulated.append((update_id, chunk_id, data, worker_ranks))
        if chunk_id == 10:
            assert not self.collection_returned.is_set(), (
                "Fast-group backward must execute before slow-group collection ends"
            )
            self.first_backward.set()
            if self.fail_backward:
                raise RuntimeError("injected backward failure")

    def finish_stream_update(self, update_id):
        assert self.collection_returned.is_set()
        self.commits.append(update_id)

    def abort_stream_update(self, update_id):
        self.aborts.append(update_id)


class _Rollout:
    def __init__(self):
        self.workers = [SimpleNamespace(ip="node"), SimpleNamespace(ip="node")]
        self.events = queue.Queue()
        self.closed = set()
        self.finished_workers = set()
        self.collecting = False
        self.pauses = 0
        self.restores = 0
        self.final_drains = 0

    def stream_topology(self):
        return [
            {"worker_rank": 0, "host": "node", "gpu_uuids": ["GPU-fast"]},
            {"worker_rank": 1, "host": "node", "gpu_uuids": ["GPU-slow"]},
        ]

    def get_version(self):
        return 7

    def begin_stream_collection(self):
        assert not self.collecting
        self.collecting = True

    def get_stream_completed(self, timeout=0.0):
        try:
            return [self.events.get(timeout=timeout)]
        except queue.Empty:
            return []

    def try_drain_stream_replica(self, worker_rank, keep_one=True):
        assert self.collecting
        if worker_rank not in self.finished_workers:
            return False
        if keep_one and len(self.closed) == len(self.workers) - 1:
            return False
        self.closed.add(worker_rank)
        return True

    def finish_stream_draining(self):
        assert self.finished_workers == {0, 1}
        self.final_drains += 1
        self.closed.update(self.finished_workers)
        return sorted(self.closed)

    def pause(self):
        self.pauses += 1

    def end_stream_collection(self):
        self.collecting = False

    def restore_stream_replicas(self):
        self.restores += 1
        self.closed.clear()


def _experiment(monkeypatch, *, fail_backward=False, wrong_membership=False):
    """The slow collector cannot finish unless early backward actually runs."""
    monkeypatch.setattr(
        "areal.trainer.stream_trainer.stats_tracker.scalar", lambda **_: None
    )
    collection_returned = threading.Event()
    actor = _Actor(collection_returned, fail_backward=fail_backward)
    rollout = _Rollout()
    fast, slow = {"sample": "fast"}, {"sample": "slow"}
    trainer = StreamTrainer(
        actor, rollout, StreamTrainerConfig(enabled=True, poll_interval=0.001)
    )

    def prepare_batch():
        rollout.finished_workers.add(0)
        rollout.events.put({"task_id": 10, "worker_rank": 0, "trajectory": fast})
        if not actor.first_backward.wait(timeout=5):
            raise AssertionError(
                "Collector received no early backward within 5 seconds"
            )
        rollout.finished_workers.add(1)
        rollout.events.put({"task_id": 20, "worker_rank": 1, "trajectory": slow})
        collection_returned.set()
        return [fast, {"sample": "impostor"} if wrong_membership else slow]

    return trainer, actor, rollout, prepare_batch, fast, slow


def test_stream_trainer_fast_backward_overlaps_collection_without_early_commit(
    monkeypatch,
):
    """An early gradient runs while rollout is live; optimizer waits for commit."""
    trainer, actor, rollout, prepare, fast, slow = _experiment(monkeypatch)

    batch = trainer.collect_and_accumulate(prepare, global_step=3)

    assert batch[0] is fast and batch[1] is slow
    assert actor.first_backward.is_set()
    assert sorted(call[1] for call in actor.accumulated) == [10, 20]
    assert actor.commits == []
    assert actor.aborts == []
    assert rollout.final_drains == 1
    assert not rollout.collecting
    trainer.commit()
    assert actor.commits == ["step-3-v7"]
    with pytest.raises(RuntimeError, match="complete streamed batch"):
        trainer.commit()
    assert len(actor.commits) == 1


def test_stream_trainer_gpu_uuid_mapping_overrides_rank_order(monkeypatch):
    """Rollout worker zero must switch actor rank one on its physical GPU."""
    trainer, actor, _, prepare, fast, _ = _experiment(monkeypatch)

    trainer.collect_and_accumulate(prepare, global_step=3)

    (fast_call,) = [call for call in actor.accumulated if call[1] == 10]
    assert fast_call[2][0] is fast
    assert fast_call[3] == [1]


def test_stream_trainer_wrong_membership_aborts_and_forbids_optimizer(monkeypatch):
    """Even equal-sized batches cannot replace a previously observed group."""
    trainer, actor, rollout, prepare, _, _ = _experiment(
        monkeypatch, wrong_membership=True
    )

    with pytest.raises(RuntimeError, match="exactly match"):
        trainer.collect_and_accumulate(prepare, global_step=3)

    assert actor.aborts == ["step-3-v7"]
    assert actor.commits == []
    assert rollout.pauses >= 1
    assert not rollout.collecting
    with pytest.raises(RuntimeError, match="complete streamed batch"):
        trainer.commit()


def test_stream_trainer_failed_backward_aborts_and_forbids_optimizer(monkeypatch):
    """A partially modified gradient buffer can never enter ordinary commit."""
    trainer, actor, rollout, prepare, _, _ = _experiment(
        monkeypatch, fail_backward=True
    )

    with pytest.raises(RuntimeError, match="injected backward failure"):
        trainer.collect_and_accumulate(prepare, global_step=3)

    assert actor.aborts == ["step-3-v7"]
    assert actor.commits == []
    assert rollout.pauses >= 1
    assert not rollout.collecting
    with pytest.raises(RuntimeError, match="complete streamed batch"):
        trainer.commit()


def test_stream_trainer_release_requires_commit_before_reopening_rollout(monkeypatch):
    """Pending gradients prevent inference admission from reopening."""
    trainer, actor, rollout, prepare, _, _ = _experiment(monkeypatch)
    trainer.collect_and_accumulate(prepare, global_step=3)
    with pytest.raises(RuntimeError, match="before a streamed update commits"):
        trainer.release_rollout()
    assert rollout.restores == 0

    trainer.commit()
    trainer.release_rollout()

    assert len(actor.commits) == 1
    assert rollout.restores == 1
    assert not rollout.closed
