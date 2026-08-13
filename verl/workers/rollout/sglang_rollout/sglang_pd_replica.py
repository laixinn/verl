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
"""SGLang hybrid Prefill-Decode (PD) disaggregation.

Implements the design in ``docs/advance/sglang_hybrid_pd_disaggregation.md``:

- ``SGLangPDReplica``: common physical-unit base (one role-specific SGLang
  server instance backed by exactly ``tp`` hybrid workers, possibly spanning
  multiple nodes).
- ``SGLangPrefillReplica`` / ``SGLangDecodeReplica``: thin role subclasses.
- ``SGLangHybridPDReplicaSet``: composite ``RolloutReplica`` consumed by
  ``LLMServerManager`` / ``CheckpointEngineManager``; owns worker placement,
  launches every physical leaf, and exposes one Ray router as the generation
  endpoint.
"""

from __future__ import annotations

import logging
import os
from contextlib import suppress
from typing import Optional

import ray
from omegaconf import DictConfig
from ray.actor import ActorHandle

from verl.single_controller.ray import RayWorkerGroup
from verl.utils.device import get_visible_devices_keyword, is_torch_npu_available
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.workers.config import RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica
from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangReplica

visible_devices_keyword = get_visible_devices_keyword()
from verl.workers.rollout.sglang_rollout.pd_topology import (
    DisaggregationRole,
    HybridPDTopology,
    PDUnitPlacement,
)
from verl.workers.rollout.sglang_rollout.sglang_pd_router import (
    PDReplicaEndpoint,
    PDReplicaRuntime,
    PDRouterFactory,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _reserve_bootstrap_port_on_worker(_worker) -> tuple[str, int]:
    """Reserve a bootstrap port in the worker process that owns the prefill node."""
    host = ray.util.get_node_ip_address().strip("[]")
    port, sock = get_free_port(host, with_alive_sock=True)
    _worker._sglang_pd_bootstrap_sock = sock
    return host, port


def _close_bootstrap_port_on_worker(_worker) -> None:
    sock = getattr(_worker, "_sglang_pd_bootstrap_sock", None)
    if sock is not None:
        sock.close()
        _worker._sglang_pd_bootstrap_sock = None


class SGLangPDReplica(SGLangReplica):
    """One physical prefill or decode SGLang instance, backed by exactly ``tp`` workers.

    Prefill and decode replicas share all implementation except role-specific
    SGLang launch arguments.
    """

    role: DisaggregationRole

    def __init__(
        self,
        placement: PDUnitPlacement,
        config: RolloutConfig,
        model_config: DictConfig,
        gpus_per_node: int = 8,
    ):
        # unit_rank must be globally unique across prefill and decode units
        # combined (prefill 0 and decode 0 would otherwise collide in
        # profiling and other replica_rank-keyed paths inherited from
        # SGLangReplica/RolloutReplica).
        super().__init__(
            replica_rank=placement.unit_rank,
            config=config,
            model_config=model_config,
            gpus_per_node=gpus_per_node,
        )
        self.placement = placement
        self.role_replica_rank = placement.role_replica_rank
        self.world_size = placement.tp_size
        self.gpus_per_replica_node = min(self.gpus_per_node, self.world_size)
        assert self.world_size % self.gpus_per_replica_node == 0
        self.nnodes = self.world_size // self.gpus_per_replica_node

        self._bootstrap_port: Optional[int] = None
        self._bootstrap_host: Optional[str] = None
        self._bootstrap_reserved = False

    async def init_hybrid_workers(self, workers: list[ActorHandle], placement: PDUnitPlacement) -> None:
        """Init this physical unit with its exact TP-sized worker slice.

        Unlike ``RolloutReplica.init_hybrid`` (which slices a full worker
        group by ``replica_rank``), the composite deployment owns slicing and
        hands each physical replica its exact worker list directly.
        """
        assert len(workers) == self.world_size, (
            f"physical PD unit {placement.unit_rank} ({placement.role.value}) needs "
            f"{self.world_size} workers, got {len(workers)}"
        )
        assert placement == self.placement
        self.rollout_mode = RolloutMode.HYBRID
        self.workers = workers
        await self.launch_servers()

    async def reserve_bootstrap_port(self) -> tuple[Optional[str], Optional[int]]:
        """Reserve a unique bootstrap socket on this unit's primary node (prefill only).

        Must be called before :meth:`launch_servers`. The reservation is owned
        by the primary node (node_rank 0) of this physical unit -- never by
        the driver process, which may run on an unrelated node and cannot
        protect a remote node from a bind race. The caller must close the
        reservation (see :meth:`close_bootstrap_reservation`) immediately
        before the SGLang bootstrap server binds.
        """
        assert self.role == DisaggregationRole.PREFILL, "only prefill units reserve a bootstrap port"
        configured_port = self.config.disaggregation.bootstrap_port
        assert len(self.workers) == self.world_size, "workers must be assigned before reserving a bootstrap port"
        if configured_port is not None:
            # Only valid when prefill_replicas == 1 (enforced by DisaggregationConfig).
            self._bootstrap_port = configured_port
            self._bootstrap_host = await self.workers[0].__ray_call__.remote(
                lambda self: ray.util.get_node_ip_address().strip("[]")
            )
            return self._bootstrap_host, self._bootstrap_port

        self._bootstrap_host, self._bootstrap_port = await self.workers[0].__ray_call__.remote(
            _reserve_bootstrap_port_on_worker
        )
        self._bootstrap_reserved = True
        return self._bootstrap_host, self._bootstrap_port

    async def close_bootstrap_reservation(self) -> None:
        """Release the reservation socket right before SGLang's bootstrap server binds."""
        if self._bootstrap_reserved:
            await self.workers[0].__ray_call__.remote(_close_bootstrap_port_on_worker)
            self._bootstrap_reserved = False

    def get_runtime(self) -> PDReplicaRuntime:
        """Return the primary leaf endpoint consumed by the Ray PD router."""
        if self._server_handle is None or self._server_address is None:
            raise RuntimeError(f"PD leaf {self.placement.primary_actor_name} has not been launched")
        is_prefill = self.role == DisaggregationRole.PREFILL
        return PDReplicaRuntime(
            endpoint=PDReplicaEndpoint(
                replica_id=self.placement.primary_actor_name,
                role=self.role,
                http_url=f"http://{self._server_address}",
                bootstrap_host=self._bootstrap_host if is_prefill else None,
                bootstrap_port=self._bootstrap_port if is_prefill else None,
            ),
            actor_handle=self._server_handle,
        )

    async def launch_servers(self) -> None:
        """Launch this physical unit's leaf server(s), reusing the symmetric single/multi-node path."""
        assert not is_torch_npu_available(check_device=False), "PD on NPU not validated"
        assert len(self.workers) == self.world_size

        extra_kwargs = {"disaggregation_role": self.role.value}
        if self.role == DisaggregationRole.PREFILL:
            extra_kwargs["disaggregation_bootstrap_port"] = self._bootstrap_port

        def name_fn(node_rank: int) -> str:
            return self.placement.actor_name_for_node(node_rank)

        # In ``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`` mode Ray does not
        # isolate workers to their individual GPU slots, so all workers see the
        # full node GPU list.  We derive the correct GPU slice for this unit
        # from its global_ranks[0], which is the first hybrid-worker rank
        # assigned to it.  The SGLang server will use
        # ``[base_gpu_id, base_gpu_id + tp_size)`` indices.
        base_gpu_id = 0
        if os.environ.get(f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword}", None):
            base_gpu_id = self.placement.begin_rank % self.gpus_per_node

        self.servers = await self._launch_server_group(
            workers=self.workers,
            base_gpu_id=base_gpu_id,
            name_fn=name_fn,
            extra_kwargs=extra_kwargs,
        )

        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )
        if self.role == DisaggregationRole.PREFILL and self._bootstrap_host is None:
            # bootstrap_port was explicitly configured (single-prefill path); host
            # is the leaf's own http address.
            self._bootstrap_host = server_address

        bootstrap_info = (
            f", bootstrap={self._bootstrap_host}:{self._bootstrap_port}"
            if self.role == DisaggregationRole.PREFILL
            else ""
        )
        logger.info(
            f"SGLangPDReplica {self.role.value} unit_rank={self.placement.unit_rank} "
            f"role_replica_rank={self.role_replica_rank} launched: address={self._server_address}{bootstrap_info}"
        )


class SGLangPrefillReplica(SGLangPDReplica):
    role = DisaggregationRole.PREFILL


class SGLangDecodeReplica(SGLangPDReplica):
    role = DisaggregationRole.DECODE


class SGLangHybridPDReplicaSet(RolloutReplica):
    """Composite hybrid PD deployment consumed by ``LLMServerManager`` / ``CheckpointEngineManager``.

    Owns worker placement and launches every physical P/D leaf. The Ray router
    is the sole generation endpoint exposed to the rollout manager.
    """

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: DictConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(
            replica_rank=replica_rank,
            config=config,
            model_config=model_config,
            gpus_per_node=gpus_per_node,
            is_reward_model=is_reward_model,
            is_teacher_model=is_teacher_model,
            name_suffix=name_suffix,
        )

        disagg = self.config.disaggregation
        assert disagg.enabled, "SGLangHybridPDReplicaSet requires rollout.disaggregation.enabled=True"

        tp_size = self.config.tensor_model_parallel_size
        # Include replica_rank as deployment_id so that actor names are unique
        # across multiple SGLangHybridPDReplicaSet instances in the same Ray
        # cluster (e.g. two replica sets both have a prefill_0 unit).
        self.topology = HybridPDTopology.build(
            tp_size=tp_size,
            prefill_replicas=disagg.prefill_replicas,
            decode_replicas=disagg.decode_replicas,
            deployment_id=str(replica_rank),
        )
        self.world_size = self.topology.world_size

        self.prefills: list[SGLangPrefillReplica] = []
        self.decodes: list[SGLangDecodeReplica] = []
        self.router = None

    async def init_hybrid(self, worker_group: RayWorkerGroup) -> None:
        """Build the topology, slice workers, launch every leaf, and start the router."""
        self.rollout_mode = RolloutMode.HYBRID

        offset = self.world_size * self.replica_rank
        sliced_workers = worker_group.workers[offset : offset + self.world_size]
        assert len(sliced_workers) == self.world_size, (
            f"worker_group has {len(worker_group.workers)} workers, cannot slice "
            f"[{offset}:{offset + self.world_size}) for hybrid PD replica_rank={self.replica_rank}"
        )

        def build_units(replica_cls, units):
            return [
                (
                    replica_cls(
                        placement=unit,
                        config=self.config,
                        model_config=self.model_config,
                        gpus_per_node=self.gpus_per_node,
                    ),
                    sliced_workers[unit.begin_rank : unit.begin_rank + unit.tp_size],
                )
                for unit in units
            ]

        prefill_pairs = build_units(SGLangPrefillReplica, self.topology.prefill_units)
        decode_pairs = build_units(SGLangDecodeReplica, self.topology.decode_units)
        launched_prefills: list[SGLangPrefillReplica] = []
        launched_decodes: list[SGLangDecodeReplica] = []
        try:

            # 4. Reserve a unique bootstrap socket on each prefill primary node.
            for prefill, unit_workers in prefill_pairs:
                prefill.workers = unit_workers
                await prefill.reserve_bootstrap_port()

            # 5. Launch prefill leaf servers. Release the reservation immediately
            # before SGLang binds the bootstrap port.
            for prefill, unit_workers in prefill_pairs:
                await prefill.close_bootstrap_reservation()
                await prefill.init_hybrid_workers(unit_workers, prefill.placement)
                launched_prefills.append(prefill)

            # 6. Launch decode leaf servers.
            for decode, unit_workers in decode_pairs:
                await decode.init_hybrid_workers(unit_workers, decode.placement)
                launched_decodes.append(decode)

            self.prefills = launched_prefills
            self.decodes = launched_decodes
            self.workers = [worker for replica in self.prefills + self.decodes for worker in replica.workers]
            self.servers = [server for replica in self.prefills + self.decodes for server in replica.servers]

            # 7-10. Register every leaf with the router and expose only that
            # logical endpoint to LLMServerManager.
            self.router = PDRouterFactory.create(self.config.disaggregation.router)
            endpoint = await self.router.start(
                [replica.get_runtime() for replica in self.prefills],
                [replica.get_runtime() for replica in self.decodes],
            )
            self._server_handle = endpoint.actor_handle
            self._server_address = endpoint.http_url or endpoint.endpoint_id
        except Exception:
            logger.exception(
                f"SGLangHybridPDReplicaSet replica_rank={self.replica_rank} init_hybrid failed; "
                f"tearing down {len(launched_prefills)} prefill(s) and {len(launched_decodes)} decode(s)"
            )
            if self.router is not None:
                with suppress(Exception):
                    await self.router.stop()
            for replica, _ in prefill_pairs + decode_pairs:
                with suppress(Exception):
                    await replica.close_bootstrap_reservation()
                for server in replica.servers:
                    with suppress(Exception):
                        ray.kill(server)
            raise

    async def launch_servers(self):
        raise NotImplementedError("SGLangHybridPDReplicaSet is initialized via init_hybrid(), not launch_servers()")
