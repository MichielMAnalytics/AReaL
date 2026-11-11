# SPDX-License-Identifier: Apache-2.0

"""Admission and observation tests; no CUDA or inference servers are required."""

import queue
import threading
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from areal.infra.controller.rollout_controller import (
    RolloutController,
    _RemoteRolloutTaskInput,
)
from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.infra.workflow_executor import _RolloutResult
from areal.utils.rollout_metrics import ConsumedMetrics


@pytest.fixture
def controller():
    """Construct only the state needed by isolated stream controller tests."""
    value = RolloutController.__new__(RolloutController)
    value.workers = [SimpleNamespace(id=f"rollout-{rank}") for rank in range(4)]
    value._worker_role = "rollout"
    value._current_worker_idx = 0
    value._stream_lock = threading.RLock()
    value._stream_collecting = False
    value._stream_events = queue.Queue()
    value._stream_active = defaultdict(int)
    value._stream_closed = set()
    value._stream_drained = set()
    value._stream_task_workers = {}
    value._stream_failure = None
    value._backend_router = None
    value._proxy_started = False
    value.scheduler = SimpleNamespace(async_call_engine=AsyncMock())
    value.consumed_metrics = ConsumedMetrics()
    return value


def test_stream_drain_active_workflow_keeps_replica_running(controller):
    """A tool phase still reserves a replica even with zero model requests."""
    controller.begin_stream_collection()
    controller._choose_worker(reserve_workflow=True)

    assert not controller.try_drain_stream_replica(0)
    controller.scheduler.async_call_engine.assert_not_called()


def test_stream_drain_idle_worker_targets_only_one_replica(controller):
    """Switching a replica must not broadcast pause or release to others."""
    controller.begin_stream_collection()

    assert controller.try_drain_stream_replica(1)

    calls = controller.scheduler.async_call_engine.call_args_list
    assert [call.kwargs["method"] for call in calls] == [
        "pause",
        "pause_generation",
        "offload",
    ]
    assert {call.kwargs["worker_id"] for call in calls} == {"rollout-1"}
    assert calls[-1].kwargs["tags"] == ["kv_cache", "weights", "cuda_graph"]
    assert controller._choose_worker()[1] == 0
    assert controller._choose_worker()[1] == 2


def test_stream_drain_last_replica_preserves_rejection_capacity(controller):
    """A final replica stays available for replacement groups after filtering."""
    controller.begin_stream_collection()
    for rank in range(3):
        assert controller.try_drain_stream_replica(rank)

    assert not controller.try_drain_stream_replica(3)
    assert controller._choose_worker()[1] == 3


def test_stream_finish_draining_releases_each_replica_once(controller):
    """A final barrier drains the last server without releasing others twice."""
    controller.begin_stream_collection()
    assert controller.try_drain_stream_replica(0)

    assert controller.finish_stream_draining() == [0, 1, 2, 3]
    assert controller.finish_stream_draining() == [0, 1, 2, 3]

    calls = controller.scheduler.async_call_engine.call_args_list
    assert len(calls) == 12
    with pytest.raises(RuntimeError, match="closed admission"):
        controller._choose_worker()


def test_stream_finish_draining_active_workflow_rejects_barrier(controller):
    """Joining generation is mandatory before closing the final replica."""
    controller.begin_stream_collection()
    controller._choose_worker(reserve_workflow=True)
    with pytest.raises(RuntimeError, match="workflows are active"):
        controller.finish_stream_draining()


def test_stream_drain_rpc_failure_fails_closed(controller):
    """A failed memory release never reopens the replica or reports success."""
    controller.begin_stream_collection()
    controller.scheduler.async_call_engine.side_effect = RuntimeError("release failed")

    with pytest.raises(RuntimeError, match="release failed"):
        controller.try_drain_stream_replica(1)

    assert 1 in controller._stream_closed
    with pytest.raises(RuntimeError, match="transition failed"):
        controller._choose_worker()
    with pytest.raises(RuntimeError, match="transition failed"):
        controller.get_stream_completed()


def test_stream_consume_selected_group_publishes_same_trajectory(controller):
    """Only consumed accepted groups are exposed, preserving tensor identity."""
    controller.begin_stream_collection()
    trajectory = {"example": object()}
    controller._stream_task_workers[11] = 2
    result = _RolloutResult(task_id=11, trajectory=trajectory, metrics={})

    assert controller._consume_rollout_result(result) is result
    (event,) = controller.get_stream_completed()

    assert event["task_id"] == 11
    assert event["worker_rank"] == 2
    assert event["trajectory"] is trajectory
    assert controller.get_stream_completed() == []


def test_stream_consume_rejected_group_does_not_publish(controller):
    """Filtering keeps the ordinary replacement path and emits no train work."""
    controller.begin_stream_collection()
    controller._stream_task_workers[11] = 2

    assert (
        controller._consume_rollout_result(
            _RolloutResult(task_id=11, trajectory=None, metrics={})
        )
        is None
    )
    assert controller.get_stream_completed() == []


def test_stream_restore_reopens_only_after_collection_ends(controller):
    """Admission restoration does not issue premature actor-memory onloads."""
    controller.begin_stream_collection()
    assert controller.try_drain_stream_replica(0)
    with pytest.raises(RuntimeError, match="during collection"):
        controller.restore_stream_replicas()
    controller.end_stream_collection()
    controller.scheduler.async_call_engine.reset_mock()

    controller.restore_stream_replicas()

    assert not controller._stream_closed
    controller.scheduler.async_call_engine.assert_not_called()


def test_stream_begin_backend_routing_rejects_unpinned_execution(controller):
    """Router leases could execute a workflow on another supposedly idle GPU."""
    controller._backend_router = object()
    with pytest.raises(ValueError, match="backend_routing"):
        controller.begin_stream_collection()


@pytest.mark.asyncio
async def test_stream_workflow_timeout_releases_admission_reservation(controller):
    """A callback timeout releases the whole-workflow reservation in finally."""
    controller.begin_stream_collection()
    controller._futures_lock = threading.Lock()
    controller._pending_futures = {}
    controller._staleness_manager = Mock()
    controller._callback_host = "localhost"
    controller._callback_port = 1234
    controller.config = SimpleNamespace(request_timeout=0.001)
    controller.scheduler.async_call_engine.return_value = 11
    task = _RemoteRolloutTaskInput(
        task_id=11,
        data={},
        workflow="example.Workflow",
        workflow_kwargs={},
        should_accept_fn=None,
    )

    assert await controller._create_submit_callback(task)() is None

    assert controller._stream_active[0] == 0
    assert not controller._pending_futures
    controller._staleness_manager.on_rollout_rejected.assert_called_once()
    with pytest.raises(RuntimeError, match="transition failed"):
        controller.try_drain_stream_replica(0)


def test_stream_topology_repeated_physical_gpu_rejects_overlap(controller):
    """Controller ranks must never be mistaken for proof of GPU placement."""
    controller._gpus_per_server = 1
    controller._collective_rpc = Mock(
        return_value=[
            {"host": "host", "gpu_uuids": ["GPU-a"]},
            {"host": "host", "gpu_uuids": ["GPU-a"]},
        ]
    )
    with pytest.raises(ValueError, match="overlap physical"):
        controller.stream_topology()


def test_stream_topology_process_tree_excludes_unrelated_cuda_process(
    monkeypatch, tmp_path
):
    """Use actual server descendants, not worker rank or visible-device indices."""
    for pid, parent in ((10, 1), (11, 10), (12, 11), (99, 1)):
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "stat").write_text(f"{pid} (name with spaces) S {parent} 0")
    monkeypatch.setattr("areal.infra.remote_inf_engine.Path", lambda _: tmp_path)
    run = Mock(
        return_value=SimpleNamespace(
            stdout=("11, GPU-first\n12, GPU-second\n99, GPU-unrelated\n")
        )
    )
    monkeypatch.setattr("areal.infra.remote_inf_engine.subprocess.run", run)
    engine = RemoteInfEngine.__new__(RemoteInfEngine)
    engine.local_server_processes = [
        SimpleNamespace(host="host", process=SimpleNamespace(pid=10, poll=lambda: None))
    ]

    assert engine.get_stream_topology() == {
        "host": "host",
        "gpu_uuids": ["GPU-first", "GPU-second"],
    }
