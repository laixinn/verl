# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GPU-free mocked launch tests for ``SGLangHybridPDReplicaSet`` / ``SGLangPDReplica``.

These tests stub out every Ray-dependent primitive (server actor creation,
bootstrap-port reservation, and the router factory) so the orchestration
logic in ``SGLangHybridPDReplicaSet.init_hybrid`` and ``SGLangPDReplica``
can be verified without a real Ray cluster, SGLang, or GPUs. Following the
design doc's "Mocked launch tests" test plan section.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Optional
from unittest.mock import patch

import pytest

# Skip the entire module when sglang's native kernel (sgl_kernel / deep_gemm)
# cannot be imported -- which happens on CPU-only machines or when the
# torch/CUDA versions don't match the pre-built wheel. This is the same
# condition that prevents importing async_sglang_server.py →
# sglang.srt.entrypoints.engine.
#
# NOTE: we cannot use pytest.importorskip here because deep_gemm raises a
# ``RuntimeError`` (not ``ImportError``) when libcudart is missing, and
# pytest.importorskip only intercepts ImportError/ModuleNotFoundError.
# Instead, we probe the import inside a broad try/except so that collection
# succeeds and pytest emits a clean SKIP rather than a collection ERROR.
try:
    import sglang.srt.entrypoints.engine  # noqa: F401  # probe only
except (ImportError, RuntimeError) as _sglang_import_err:
    pytest.skip(
        f"sglang kernel not importable on this machine "
        f"(sgl_kernel/CUDA mismatch: {_sglang_import_err}); "
        "skipping PD replica launch tests",
        allow_module_level=True,
    )

from verl.workers.config import DisaggregationConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.sglang_rollout.pd_topology import DisaggregationRole, PDUnitPlacement
from verl.workers.rollout.sglang_rollout.sglang_pd_router import PDReplicaEndpoint, PDReplicaRuntime, RolloutEndpoint


class FakeWorker:
    """Stand-in for a hybrid worker ``ActorHandle``; identity is all that matters here."""

    def __init__(self, idx: int):
        self.idx = idx

    def __repr__(self):
        return f"FakeWorker({self.idx})"


class FakeWorkerGroup:
    """Minimal stand-in for ``RayWorkerGroup`` -- only ``.workers`` is read by ``init_hybrid``."""

    def __init__(self, world_size: int):
        self.workers = [FakeWorker(i) for i in range(world_size)]
        self.world_size = world_size


@dataclass
class FakeServerHandle:
    """Stand-in for a launched ``SGLangHttpServer`` Ray actor handle."""

    name: str
    role: str
    address: str = "10.0.0.1"
    port: int = 30000

    async def get_server_address(self):
        return self.address, self.port


class FakeRouterController:
    """Stand-in ``PDRouterController`` recording every call for assertions."""

    instances: list["FakeRouterController"] = []

    def __init__(self, config=None):
        self.config = config
        self.started_with: Optional[tuple[list[PDReplicaRuntime], list[PDReplicaRuntime]]] = None
        self.stopped = False
        FakeRouterController.instances.append(self)

    async def start(self, prefills, decodes) -> RolloutEndpoint:
        self.started_with = (list(prefills), list(decodes))
        return RolloutEndpoint(endpoint_id="fake_router", actor_handle=self, http_url=None)

    async def stop(self) -> None:
        self.stopped = True


def _make_pd_rollout_config(*, tp_size: int, prefill_replicas: int, decode_replicas: int) -> RolloutConfig:
    return RolloutConfig(
        name="sglang",
        tensor_model_parallel_size=tp_size,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        disaggregation=DisaggregationConfig(
            enabled=True,
            prefill_replicas=prefill_replicas,
            decode_replicas=decode_replicas,
        ),
    )


@pytest.fixture(autouse=True)
def _reset_fake_router_instances():
    FakeRouterController.instances = []
    yield
    FakeRouterController.instances = []


def _patch_pd_replica_module():
    """Return the ``sglang_pd_replica`` module, importing lazily so pytest collection
    doesn't require sglang/vllm to be importable until this test module actually runs."""
    from verl.workers.rollout.sglang_rollout import sglang_pd_replica as mod

    return mod


def _install_fake_launch(mod, launched: list[dict]):
    """Patch ``SGLangPDReplica.launch_servers``/bootstrap methods with Ray-free fakes.

    Records every launch call (role, unit_rank, workers) into ``launched`` so tests can
    assert exact worker slicing without needing real Ray actors or SGLang processes.
    """

    async def fake_launch_servers(self):
        assert len(self.workers) == self.world_size
        handle = FakeServerHandle(name=self.placement.primary_actor_name, role=self.role.value)
        self.servers = [handle]
        self._server_handle = handle
        self._server_address = f"{handle.address}:{handle.port}"
        if self.role == DisaggregationRole.PREFILL and self._bootstrap_host is None:
            self._bootstrap_host = handle.address
        launched.append(
            {
                "role": self.role.value,
                "unit_rank": self.placement.unit_rank,
                "role_replica_rank": self.role_replica_rank,
                "workers": list(self.workers),
            }
        )

    async def fake_reserve_bootstrap_port(self):
        assert self.role == DisaggregationRole.PREFILL
        configured_port = self.config.disaggregation.bootstrap_port
        if configured_port is not None:
            self._bootstrap_port = configured_port
            self._bootstrap_host = "10.0.0.1"
        else:
            self._bootstrap_port = 20000 + self.placement.role_replica_rank
            self._bootstrap_host = f"10.0.0.{self.placement.role_replica_rank + 1}"
        return self._bootstrap_host, self._bootstrap_port

    return (
        patch.object(mod.SGLangPDReplica, "launch_servers", fake_launch_servers),
        patch.object(mod.SGLangPDReplica, "reserve_bootstrap_port", fake_reserve_bootstrap_port),
    )


# ---------------------------------------------------------------------------
# Worker slicing correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tp_size,prefill_replicas,decode_replicas",
    [(8, 1, 1), (8, 2, 6), (4, 2, 2), (1, 7, 6)],
)
async def test_worker_slices_passed_to_each_physical_replica_are_correct(tp_size, prefill_replicas, decode_replicas):
    mod = _patch_pd_replica_module()
    cfg = _make_pd_rollout_config(tp_size=tp_size, prefill_replicas=prefill_replicas, decode_replicas=decode_replicas)
    world_size = (prefill_replicas + decode_replicas) * tp_size
    wg = FakeWorkerGroup(world_size)

    launched: list[dict] = []
    launch_patch, bootstrap_patch = _install_fake_launch(mod, launched)
    with launch_patch, bootstrap_patch, patch.object(mod, "PDRouterFactory") as fake_factory:
        fake_factory.create.return_value = FakeRouterController()
        replica_set = mod.SGLangHybridPDReplicaSet(
            replica_rank=0, config=cfg, model_config=None, gpus_per_node=8
        )
        await replica_set.init_hybrid(wg)

    # Every physical unit received exactly tp_size workers, matching the topology's
    # contiguous slice, and no worker is shared or skipped.
    assert len(launched) == prefill_replicas + decode_replicas
    seen_idxs: set[int] = set()
    for i, entry in enumerate(launched):
        assert len(entry["workers"]) == tp_size
        idxs = {w.idx for w in entry["workers"]}
        assert idxs == set(range(i * tp_size, (i + 1) * tp_size))
        seen_idxs |= idxs
    assert seen_idxs == set(range(world_size))

    # Prefill units precede decode units (canonical topology ordering).
    assert [entry["role"] for entry in launched[:prefill_replicas]] == ["prefill"] * prefill_replicas
    assert [entry["role"] for entry in launched[prefill_replicas:]] == ["decode"] * decode_replicas


@pytest.mark.asyncio
async def test_replica_rank_offset_slices_correct_window():
    """A non-zero ``replica_rank`` (multiple hybrid PD deployments sharing one
    worker_group) must slice its own contiguous window, not always start at 0."""
    mod = _patch_pd_replica_module()
    cfg = _make_pd_rollout_config(tp_size=2, prefill_replicas=1, decode_replicas=1)
    # Two PD deployments' worth of workers; replica_rank=1 should use the second half.
    wg = FakeWorkerGroup(world_size=8)

    launched: list[dict] = []
    launch_patch, bootstrap_patch = _install_fake_launch(mod, launched)
    with launch_patch, bootstrap_patch, patch.object(mod, "PDRouterFactory") as fake_factory:
        fake_factory.create.return_value = FakeRouterController()
        replica_set = mod.SGLangHybridPDReplicaSet(replica_rank=1, config=cfg, model_config=None, gpus_per_node=8)
        await replica_set.init_hybrid(wg)

    all_idxs = sorted(w.idx for entry in launched for w in entry["workers"])
    assert all_idxs == [4, 5, 6, 7]


# ---------------------------------------------------------------------------
# Shared launch path / role-specific arguments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prefill_and_decode_use_the_same_common_launch_path():
    """Both roles go through ``SGLangPDReplica.launch_servers`` -- same patched
    function services both, proving there is exactly one launch implementation."""
    mod = _patch_pd_replica_module()
    cfg = _make_pd_rollout_config(tp_size=2, prefill_replicas=1, decode_replicas=1)
    wg = FakeWorkerGroup(world_size=4)

    launch_calls: list[str] = []

    async def fake_launch_servers(self):
        launch_calls.append(self.role.value)
        handle = FakeServerHandle(name=self.placement.primary_actor_name, role=self.role.value)
        self.servers = [handle]
        self._server_handle = handle
        self._server_address = f"{handle.address}:{handle.port}"

    async def fake_reserve(self):
        self._bootstrap_port = 21000
        self._bootstrap_host = "10.0.0.9"
        return self._bootstrap_host, self._bootstrap_port

    with (
        patch.object(mod.SGLangPDReplica, "launch_servers", fake_launch_servers),
        patch.object(mod.SGLangPDReplica, "reserve_bootstrap_port", fake_reserve),
        patch.object(mod, "PDRouterFactory") as fake_factory,
    ):
        fake_factory.create.return_value = FakeRouterController()
        replica_set = mod.SGLangHybridPDReplicaSet(replica_rank=0, config=cfg, model_config=None, gpus_per_node=8)
        await replica_set.init_hybrid(wg)

    assert launch_calls == ["prefill", "decode"]


# ---------------------------------------------------------------------------
# Bootstrap-port reservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_each_prefill_reserves_a_unique_bootstrap_port():
    mod = _patch_pd_replica_module()
    cfg = _make_pd_rollout_config(tp_size=1, prefill_replicas=3, decode_replicas=2)
    wg = FakeWorkerGroup(world_size=5)

    launched: list[dict] = []
    launch_patch, bootstrap_patch = _install_fake_launch(mod, launched)
    with launch_patch, bootstrap_patch, patch.object(mod, "PDRouterFactory") as fake_factory:
        fake_factory.create.return_value = FakeRouterController()
        replica_set = mod.SGLangHybridPDReplicaSet(replica_rank=0, config=cfg, model_config=None, gpus_per_node=8)
        await replica_set.init_hybrid(wg)

    bootstrap_ports = {p.get_runtime().endpoint.bootstrap_port for p in replica_set.prefills}
    assert len(bootstrap_ports) == 3  # all distinct


@pytest.mark.asyncio
async def test_single_prefill_preserves_explicit_bootstrap_port():
    mod = _patch_pd_replica_module()
    cfg = _make_pd_rollout_config(tp_size=2, prefill_replicas=1, decode_replicas=1)
    object.__setattr__(cfg.disaggregation, "bootstrap_port", 15000)
    wg = FakeWorkerGroup(world_size=4)

    launched: list[dict] = []
    launch_patch, bootstrap_patch = _install_fake_launch(mod, launched)
    with launch_patch, bootstrap_patch, patch.object(mod, "PDRouterFactory") as fake_factory:
        fake_factory.create.return_value = FakeRouterController()
        replica_set = mod.SGLangHybridPDReplicaSet(replica_rank=0, config=cfg, model_config=None, gpus_per_node=8)
        await replica_set.init_hybrid(wg)

    assert replica_set.prefills[0].get_runtime().endpoint.bootstrap_port == 15000


# ---------------------------------------------------------------------------
# Router registration + composite view
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_receives_all_prefill_and_decode_runtimes():
    mod = _patch_pd_replica_module()
    cfg = _make_pd_rollout_config(tp_size=1, prefill_replicas=2, decode_replicas=3)
    wg = FakeWorkerGroup(world_size=5)

    launched: list[dict] = []
    launch_patch, bootstrap_patch = _install_fake_launch(mod, launched)
    with launch_patch, bootstrap_patch, patch.object(mod, "PDRouterFactory") as fake_factory:
        fake_router = FakeRouterController()
        fake_factory.create.return_value = fake_router
        replica_set = mod.SGLangHybridPDReplicaSet(replica_rank=0, config=cfg, model_config=None, gpus_per_node=8)
        await replica_set.init_hybrid(wg)

    prefill_runtimes, decode_runtimes = fake_router.started_with
    assert len(prefill_runtimes) == 2
    assert len(decode_runtimes) == 3
    assert all(r.endpoint.role == DisaggregationRole.PREFILL for r in prefill_runtimes)
    assert all(r.endpoint.role == DisaggregationRole.DECODE for r in decode_runtimes)


@pytest.mark.asyncio
async def test_only_router_endpoint_exposed_as_compatibility_fields():
    mod = _patch_pd_replica_module()
    cfg = _make_pd_rollout_config(tp_size=2, prefill_replicas=1, decode_replicas=1)
    wg = FakeWorkerGroup(world_size=4)

    launched: list[dict] = []
    launch_patch, bootstrap_patch = _install_fake_launch(mod, launched)
    with launch_patch, bootstrap_patch, patch.object(mod, "PDRouterFactory") as fake_factory:
        fake_router = FakeRouterController()
        fake_factory.create.return_value = fake_router
        replica_set = mod.SGLangHybridPDReplicaSet(replica_rank=0, config=cfg, model_config=None, gpus_per_node=8)
        await replica_set.init_hybrid(wg)

    assert replica_set._server_handle is fake_router
    assert replica_set._server_address == "fake_router"


@pytest.mark.asyncio
async def test_flattened_workers_and_servers_views():
    mod = _patch_pd_replica_module()
    cfg = _make_pd_rollout_config(tp_size=2, prefill_replicas=1, decode_replicas=2)
    wg = FakeWorkerGroup(world_size=6)

    launched: list[dict] = []
    launch_patch, bootstrap_patch = _install_fake_launch(mod, launched)
    with launch_patch, bootstrap_patch, patch.object(mod, "PDRouterFactory") as fake_factory:
        fake_factory.create.return_value = FakeRouterController()
        replica_set = mod.SGLangHybridPDReplicaSet(replica_rank=0, config=cfg, model_config=None, gpus_per_node=8)
        await replica_set.init_hybrid(wg)

    assert len(replica_set.workers) == 6
    assert [w.idx for w in replica_set.workers] == list(range(6))
    assert len(replica_set.servers) == 3  # 1 prefill + 2 decode leaf servers


@pytest.mark.asyncio
async def test_rollout_mode_set_to_hybrid_on_every_physical_replica():
    mod = _patch_pd_replica_module()
    cfg = _make_pd_rollout_config(tp_size=2, prefill_replicas=1, decode_replicas=1)
    wg = FakeWorkerGroup(world_size=4)

    launched: list[dict] = []
    launch_patch, bootstrap_patch = _install_fake_launch(mod, launched)
    with launch_patch, bootstrap_patch, patch.object(mod, "PDRouterFactory") as fake_factory:
        fake_factory.create.return_value = FakeRouterController()
        replica_set = mod.SGLangHybridPDReplicaSet(replica_rank=0, config=cfg, model_config=None, gpus_per_node=8)
        await replica_set.init_hybrid(wg)

    assert replica_set.rollout_mode == RolloutMode.HYBRID
    for replica in replica_set.prefills + replica_set.decodes:
        assert replica.rollout_mode == RolloutMode.HYBRID


# ---------------------------------------------------------------------------
# Failure cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_launch_failure_cleans_up_already_created_actors_and_sockets():
    mod = _patch_pd_replica_module()
    cfg = _make_pd_rollout_config(tp_size=1, prefill_replicas=1, decode_replicas=2)
    wg = FakeWorkerGroup(world_size=3)

    closed_sockets: list[int] = []
    killed_servers: list[str] = []

    async def fake_reserve(self):
        self._bootstrap_port = 22000
        self._bootstrap_host = "10.0.0.5"
        return self._bootstrap_host, self._bootstrap_port

    async def fake_close_bootstrap(self):
        if self._bootstrap_port is not None:
            closed_sockets.append(self._bootstrap_port)
        self._bootstrap_reserved = False

    call_count = {"decode": 0}

    async def fake_launch_servers(self):
        if self.role == DisaggregationRole.DECODE:
            call_count["decode"] += 1
            if call_count["decode"] == 2:
                raise RuntimeError("simulated decode launch failure")
        handle = FakeServerHandle(name=self.placement.primary_actor_name, role=self.role.value)
        self.servers = [handle]
        self._server_handle = handle
        self._server_address = f"{handle.address}:{handle.port}"

    with (
        patch.object(mod.SGLangPDReplica, "launch_servers", fake_launch_servers),
        patch.object(mod.SGLangPDReplica, "reserve_bootstrap_port", fake_reserve),
        patch.object(mod.SGLangPDReplica, "close_bootstrap_reservation", fake_close_bootstrap),
        patch.object(mod, "PDRouterFactory") as fake_factory,
        patch.object(mod.ray, "kill", lambda server: killed_servers.append(server.name)),
    ):
        fake_factory.create.return_value = FakeRouterController()
        replica_set = mod.SGLangHybridPDReplicaSet(replica_rank=0, config=cfg, model_config=None, gpus_per_node=8)
        with pytest.raises(RuntimeError, match="simulated decode launch failure"):
            await replica_set.init_hybrid(wg)

    # Leaves launched before the failure must be torn down, and the router
    # must not be created until every leaf is ready.
    assert replica_set.topology.prefill_units[0].primary_actor_name in killed_servers
    assert 22000 in closed_sockets
    fake_factory.create.assert_not_called()


@pytest.mark.asyncio
async def test_launch_failure_before_router_creation_does_not_touch_router():
    """If every leaf fails before step 8 (router construction), ``self.router``
    stays ``None`` and no router-stop call is attempted."""
    mod = _patch_pd_replica_module()
    cfg = _make_pd_rollout_config(tp_size=1, prefill_replicas=1, decode_replicas=1)
    wg = FakeWorkerGroup(world_size=2)

    async def fake_reserve(self):
        self._bootstrap_port = 23000
        self._bootstrap_host = "10.0.0.6"
        return self._bootstrap_host, self._bootstrap_port

    async def failing_launch_servers(self):
        raise RuntimeError("boom")

    with (
        patch.object(mod.SGLangPDReplica, "launch_servers", failing_launch_servers),
        patch.object(mod.SGLangPDReplica, "reserve_bootstrap_port", fake_reserve),
        patch.object(mod, "PDRouterFactory") as fake_factory,
    ):
        replica_set = mod.SGLangHybridPDReplicaSet(replica_rank=0, config=cfg, model_config=None, gpus_per_node=8)
        with pytest.raises(RuntimeError, match="boom"):
            await replica_set.init_hybrid(wg)

    assert replica_set.router is None
    fake_factory.create.assert_not_called()


# ---------------------------------------------------------------------------
# CUDA runtime library discovery
# ---------------------------------------------------------------------------


def test_cu12_runtime_namespace_package_prepends_library_path(monkeypatch, tmp_path):
    """The CUDA runtime wheel is an implicit namespace package with no ``__file__``."""
    from verl.workers.rollout.sglang_rollout import async_sglang_server as server_mod

    missing_package_path = tmp_path / "missing"
    runtime_package_path = tmp_path / "cuda_runtime"
    runtime_lib = runtime_package_path / "lib"
    runtime_lib.mkdir(parents=True)
    (runtime_lib / "libcudart.so.12").touch()

    fake_cuda_runtime = ModuleType("nvidia.cuda_runtime")
    fake_cuda_runtime.__file__ = None
    fake_cuda_runtime.__path__ = [str(missing_package_path), str(runtime_package_path)]
    monkeypatch.setitem(sys.modules, "nvidia.cuda_runtime", fake_cuda_runtime)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")

    server = object.__new__(server_mod.SGLangHttpServer)
    server._prepend_cu12_lib_to_ld_library_path()
    expected = f"{runtime_lib}{os.pathsep}/existing"
    assert os.environ["LD_LIBRARY_PATH"] == expected

    server._prepend_cu12_lib_to_ld_library_path()
    assert os.environ["LD_LIBRARY_PATH"] == expected


# ---------------------------------------------------------------------------
# Multi-node physical units (node-rank / master-address plumbing)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_node_unit_preserves_nnodes_and_gpus_per_replica_node():
    """A tp=16 unit on gpus_per_node=8 spans exactly 2 nodes; SGLangPDReplica
    must compute nnodes/gpus_per_replica_node consistently with SGLangReplica."""
    mod = _patch_pd_replica_module()
    placement = PDUnitPlacement(
        unit_rank=0,
        role=DisaggregationRole.PREFILL,
        role_replica_rank=0,
        global_ranks=tuple(range(16)),
        tp_size=16,
        primary_actor_name="sglang_pd_hybrid_prefill_0_node_0",
    )
    cfg = _make_pd_rollout_config(tp_size=16, prefill_replicas=1, decode_replicas=1)
    replica = mod.SGLangPrefillReplica(placement=placement, config=cfg, model_config=None, gpus_per_node=8)
    assert replica.nnodes == 2
    assert replica.gpus_per_replica_node == 8
    assert replica.world_size == 16


def test_get_runtime_only_prefill_exposes_bootstrap_metadata():
    mod = _patch_pd_replica_module()
    placement = PDUnitPlacement(
        unit_rank=1,
        role=DisaggregationRole.DECODE,
        role_replica_rank=0,
        global_ranks=(2, 3),
        tp_size=2,
        primary_actor_name="sglang_pd_hybrid_decode_0_node_0",
    )
    cfg = _make_pd_rollout_config(tp_size=2, prefill_replicas=1, decode_replicas=1)
    replica = mod.SGLangDecodeReplica(placement=placement, config=cfg, model_config=None, gpus_per_node=8)
    replica._server_address = "10.0.0.1:1234"
    replica._server_handle = FakeServerHandle(name="x", role="decode")
    replica._bootstrap_host = "10.0.0.1"  # decode never advertises bootstrap metadata
    replica._bootstrap_port = 9999

    runtime = replica.get_runtime()
    assert isinstance(runtime, PDReplicaRuntime)
    assert isinstance(runtime.endpoint, PDReplicaEndpoint)
    assert runtime.endpoint.bootstrap_host is None
    assert runtime.endpoint.bootstrap_port is None
    assert runtime.endpoint.replica_id == "sglang_pd_hybrid_decode_0_node_0"
