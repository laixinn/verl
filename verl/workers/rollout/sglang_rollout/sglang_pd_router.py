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
"""Router abstraction for SGLang hybrid Prefill-Decode disaggregation."""

from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import ray
from ray.actor import ActorHandle

from verl.workers.config.disaggregation import PDRouterConfig
from verl.workers.rollout.replica import RolloutMode, TokenOutput
from verl.workers.rollout.sglang_rollout.pd_topology import DisaggregationRole

logger = logging.getLogger(__file__)


@dataclass(frozen=True, slots=True)
class PDReplicaEndpoint:
    """Transport-neutral description of one physical P/D replica."""

    replica_id: str
    role: DisaggregationRole
    http_url: str
    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None


@dataclass(frozen=True, slots=True)
class PDReplicaRuntime:
    """Portable endpoint metadata plus the Ray handle used by verl's router."""

    endpoint: PDReplicaEndpoint
    actor_handle: ActorHandle


@dataclass(frozen=True, slots=True)
class RolloutEndpoint:
    """Logical generation endpoint exposed by a router controller."""

    endpoint_id: str
    actor_handle: Optional[ActorHandle] = None
    http_url: Optional[str] = None


class PDRouterController(Protocol):
    """Control-plane contract for the hybrid PD router."""

    async def start(
        self,
        prefills: list[PDReplicaRuntime],
        decodes: list[PDReplicaRuntime],
    ) -> RolloutEndpoint: ...

    async def stop(self) -> None: ...


class StandalonePDRouterController(PDRouterController, Protocol):
    """Additional admission controls owned only by the standalone PD router."""

    async def quiesce(self) -> None: ...

    async def resume(self) -> None: ...

    async def wait_idle(self) -> None: ...


class SGLangPDRouter:
    """Ray actor that pairs one prefill and one decode leaf per request."""

    def __init__(self, config: PDRouterConfig):
        self.config = config
        self._backends: dict[DisaggregationRole, dict[str, PDReplicaRuntime]] = {
            DisaggregationRole.PREFILL: {},
            DisaggregationRole.DECODE: {},
        }
        self._inflight: dict[DisaggregationRole, dict[str, int]] = {
            DisaggregationRole.PREFILL: {},
            DisaggregationRole.DECODE: {},
        }
        self._unhealthy: set[str] = set()
        self._pairs: dict[str, tuple[str, str]] = {}

    def register_backends(self, prefills: list[PDReplicaRuntime], decodes: list[PDReplicaRuntime]) -> None:
        for role, runtimes in (
            (DisaggregationRole.PREFILL, prefills),
            (DisaggregationRole.DECODE, decodes),
        ):
            self._backends[role] = {runtime.endpoint.replica_id: runtime for runtime in runtimes}
            self._inflight[role] = dict.fromkeys(self._backends[role], 0)
        self._unhealthy.clear()

    def get_status(self) -> dict[str, Any]:
        return {
            "prefill_inflight": dict(self._inflight[DisaggregationRole.PREFILL]),
            "decode_inflight": dict(self._inflight[DisaggregationRole.DECODE]),
            "unhealthy": sorted(self._unhealthy),
            "num_pairs_inflight": len(self._pairs),
        }

    def mark_unhealthy(self, replica_id: str) -> None:
        self._unhealthy.add(replica_id)

    def _select(self, role: DisaggregationRole, policy: str) -> PDReplicaRuntime:
        inflight = self._inflight[role]
        candidates = [replica_id for replica_id in inflight if replica_id not in self._unhealthy]
        if not candidates:
            raise RuntimeError(f"no healthy {role.value} backend available")
        if policy == "least_inflight":
            minimum = min(inflight[replica_id] for replica_id in candidates)
            candidates = [replica_id for replica_id in candidates if inflight[replica_id] == minimum]
        replica_id = secrets.choice(candidates)
        return self._backends[role][replica_id]

    def _acquire_pair(self, request_id: str) -> tuple[PDReplicaRuntime, PDReplicaRuntime]:
        if request_id in self._pairs:
            raise RuntimeError(f"request_id {request_id!r} is already active")

        prefill = self._select(DisaggregationRole.PREFILL, self.config.prefill_policy)
        decode = self._select(DisaggregationRole.DECODE, self.config.decode_policy)
        prefill_id = prefill.endpoint.replica_id
        decode_id = decode.endpoint.replica_id
        self._inflight[DisaggregationRole.PREFILL][prefill_id] += 1
        self._inflight[DisaggregationRole.DECODE][decode_id] += 1
        self._pairs[request_id] = (prefill_id, decode_id)
        return prefill, decode

    def _release_pair(self, request_id: str) -> None:
        pair = self._pairs.pop(request_id, None)
        if pair is None:
            return
        for role, replica_id in zip(DisaggregationRole, pair, strict=True):
            role_inflight = self._inflight[role]
            if replica_id in role_inflight:
                role_inflight[replica_id] = max(0, role_inflight[replica_id] - 1)

    async def generate(self, request_id: str, **request: Any) -> TokenOutput:
        prefill, decode = self._acquire_pair(request_id)
        bootstrap = {
            "bootstrap_host": prefill.endpoint.bootstrap_host,
            "bootstrap_port": prefill.endpoint.bootstrap_port,
            "bootstrap_room": secrets.randbits(63),
        }
        refs = (
            prefill.actor_handle.generate.remote(request_id=f"{request_id}:prefill", **request, **bootstrap),
            decode.actor_handle.generate.remote(request_id=f"{request_id}:decode", **request, **bootstrap),
        )

        try:
            _, decode_output = await asyncio.gather(*refs)
            return decode_output
        except BaseException:
            for ref in refs:
                with suppress(Exception):
                    ray.cancel(ref, force=False)
            await asyncio.gather(*refs, return_exceptions=True)
            raise
        finally:
            self._release_pair(request_id)


class SGLangStandalonePDRouter(SGLangPDRouter):
    """Standalone router with admission control and in-flight drain tracking."""

    def __init__(self, config: PDRouterConfig):
        super().__init__(config)
        self._admission_open = asyncio.Event()
        self._admission_open.set()
        self._idle = asyncio.Event()
        self._idle.set()

    async def generate(self, request_id: str, **request: Any) -> TokenOutput:
        """Wait for temporary weight synchronization to finish before admitting new work."""
        await self._admission_open.wait()
        return await super().generate(request_id, **request)

    def _acquire_pair(self, request_id: str) -> tuple[PDReplicaRuntime, PDReplicaRuntime]:
        pair = super()._acquire_pair(request_id)
        self._idle.clear()
        return pair

    def _release_pair(self, request_id: str) -> None:
        super()._release_pair(request_id)
        if not self._pairs:
            self._idle.set()

    async def quiesce(self) -> None:
        self._admission_open.clear()

    async def resume(self) -> None:
        self._admission_open.set()

    async def wait_idle(self, timeout_s: float = 10.0) -> None:
        if self._idle.is_set():
            return
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout_s)
        except TimeoutError:
            logger.warning(
                "Standalone PD router did not drain %d active request pairs within %.1f seconds",
                len(self._pairs),
                timeout_s,
            )
            raise


class RayPDRouterController:
    """Controller for a Ray-hosted hybrid :class:`SGLangPDRouter`."""

    router_cls = SGLangPDRouter

    def __init__(self, config: PDRouterConfig, router_id: str):
        self.config = config
        self.router_id = router_id
        self._actor: Optional[ActorHandle] = None

    async def start(
        self,
        prefills: list[PDReplicaRuntime],
        decodes: list[PDReplicaRuntime],
    ) -> RolloutEndpoint:
        if self._actor is None:
            self._actor = (
                ray.remote(self.router_cls)
                .options(name=self.router_id, max_concurrency=10000)
                .remote(self.config)
            )
        await self._actor.register_backends.remote(prefills, decodes)
        return RolloutEndpoint(endpoint_id=self.router_id, actor_handle=self._actor)

    async def stop(self) -> None:
        if self._actor is not None:
            ray.kill(self._actor)
            self._actor = None


class RayStandalonePDRouterController(RayPDRouterController):
    """Controller for a Ray-hosted standalone :class:`SGLangStandalonePDRouter`."""

    router_cls = SGLangStandalonePDRouter

    def _require_actor(self) -> ActorHandle:
        if self._actor is None:
            raise RuntimeError("standalone PD router has not been started")
        return self._actor

    async def quiesce(self) -> None:
        await self._require_actor().quiesce.remote()

    async def resume(self) -> None:
        await self._require_actor().resume.remote()

    async def wait_idle(self, timeout_s: float = 10.0) -> None:
        await self._require_actor().wait_idle.remote(timeout_s)


class PDRouterFactory:
    """Build the configured router controller."""

    @staticmethod
    def create(
        config: PDRouterConfig,
        rollout_mode: RolloutMode,
        deployment_id: str,
    ) -> PDRouterController | StandalonePDRouterController:
        if config.backend == "ray":
            router_id = f"sglang_pd_router_{rollout_mode.value}_{deployment_id}"
            if rollout_mode == RolloutMode.HYBRID:
                return RayPDRouterController(config, router_id)
            if rollout_mode == RolloutMode.STANDALONE:
                return RayStandalonePDRouterController(config, router_id)
            raise ValueError(f"Unsupported PD router rollout mode: {rollout_mode!r}")
        if config.backend == "sglang":
            raise NotImplementedError(
                "Native sglang_router integration is not implemented yet; "
                "use disaggregation.router.backend='ray'."
            )
        raise ValueError(f"Unknown PD router backend: {config.backend!r}")
