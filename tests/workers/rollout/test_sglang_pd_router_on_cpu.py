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
"""GPU-free tests for the SGLang-only hybrid PD router."""

from __future__ import annotations

import asyncio

import pytest

from verl.workers.config.disaggregation import PDRouterConfig
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.sglang_rollout.pd_topology import DisaggregationRole
from verl.workers.rollout.sglang_rollout.sglang_pd_router import (
    PDReplicaEndpoint,
    PDReplicaRuntime,
    PDRouterFactory,
    SGLangPDRouter,
)


class _FakeRemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class FakeLeafActor:
    def __init__(self, replica_id: str, role: str, fail: bool = False, delay: float = 0.0):
        self.replica_id = replica_id
        self.role = role
        self.fail = fail
        self.delay = delay
        self.calls: list[dict] = []
        self.generate = _FakeRemoteMethod(self._generate)

    async def _generate(self, request_id: str, **kwargs) -> TokenOutput:
        self.calls.append({"request_id": request_id, **kwargs})
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError(f"{self.replica_id} intentionally failed")
        return TokenOutput(token_ids=[1, 2, 3], extra_fields={"served_by": self.replica_id, "role": self.role})


def _make_runtime(replica_id: str, role: DisaggregationRole, fail: bool = False, delay: float = 0.0):
    actor = FakeLeafActor(replica_id, role.value, fail=fail, delay=delay)
    endpoint = PDReplicaEndpoint(
        replica_id=replica_id,
        role=role,
        http_url=f"http://fake/{replica_id}",
        bootstrap_host="127.0.0.1" if role == DisaggregationRole.PREFILL else None,
        bootstrap_port=12345 if role == DisaggregationRole.PREFILL else None,
    )
    return PDReplicaRuntime(endpoint=endpoint, actor_handle=actor)


def _make_router(prefills, decodes, config: PDRouterConfig | None = None) -> SGLangPDRouter:
    router = SGLangPDRouter(config or PDRouterConfig())
    router.register_backends(prefills, decodes)
    return router


@pytest.mark.asyncio
async def test_router_pairs_one_prefill_and_decode_with_shared_bootstrap_metadata():
    prefills = [_make_runtime("p0", DisaggregationRole.PREFILL)]
    decodes = [_make_runtime("d0", DisaggregationRole.DECODE)]
    router = _make_router(prefills, decodes)

    output = await router.generate(request_id="req-1", prompt_ids=[1, 2, 3])

    assert output.extra_fields == {"served_by": "d0", "role": "decode"}
    p_call = prefills[0].actor_handle.calls[0]
    d_call = decodes[0].actor_handle.calls[0]
    assert p_call["bootstrap_room"] == d_call["bootstrap_room"]
    assert p_call["bootstrap_host"] == d_call["bootstrap_host"] == "127.0.0.1"
    assert p_call["bootstrap_port"] == d_call["bootstrap_port"] == 12345
    assert p_call["request_id"] == "req-1:prefill"
    assert d_call["request_id"] == "req-1:decode"


@pytest.mark.asyncio
async def test_inflight_counters_are_independent_and_released():
    prefills = [_make_runtime("p0", DisaggregationRole.PREFILL, delay=0.05)]
    decodes = [_make_runtime("d0", DisaggregationRole.DECODE, delay=0.05)]
    router = _make_router(prefills, decodes)

    task = asyncio.create_task(router.generate(request_id="req-1", prompt_ids=[1]))
    await asyncio.sleep(0.01)
    status = router.get_status()
    assert status["prefill_inflight"] == {"p0": 1}
    assert status["decode_inflight"] == {"d0": 1}

    await task
    status = router.get_status()
    assert status["prefill_inflight"] == {"p0": 0}
    assert status["decode_inflight"] == {"d0": 0}
    assert status["num_pairs_inflight"] == 0


@pytest.mark.asyncio
async def test_least_inflight_uses_unequal_backend_pools():
    prefills = [_make_runtime("p0", DisaggregationRole.PREFILL)]
    decodes = [
        _make_runtime("d0", DisaggregationRole.DECODE, delay=0.05),
        _make_runtime("d1", DisaggregationRole.DECODE, delay=0.001),
    ]
    router = _make_router(prefills, decodes)

    await asyncio.gather(
        *[router.generate(request_id=f"req-{index}", prompt_ids=[1]) for index in range(6)]
    )

    assert sum(len(runtime.actor_handle.calls) for runtime in decodes) == 6
    assert all(runtime.actor_handle.calls for runtime in decodes)


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_role", [DisaggregationRole.PREFILL, DisaggregationRole.DECODE])
async def test_failure_releases_pair_state(failing_role):
    prefills = [
        _make_runtime(
            "p0",
            DisaggregationRole.PREFILL,
            fail=failing_role == DisaggregationRole.PREFILL,
            delay=0.02,
        )
    ]
    decodes = [
        _make_runtime(
            "d0",
            DisaggregationRole.DECODE,
            fail=failing_role == DisaggregationRole.DECODE,
            delay=0.02,
        )
    ]
    router = _make_router(prefills, decodes)

    with pytest.raises(RuntimeError, match="intentionally failed"):
        await router.generate(request_id="req-1", prompt_ids=[1])

    status = router.get_status()
    assert status["prefill_inflight"] == {"p0": 0}
    assert status["decode_inflight"] == {"d0": 0}
    assert status["num_pairs_inflight"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_role", [DisaggregationRole.PREFILL, DisaggregationRole.DECODE])
async def test_empty_role_pool_rejects_request(missing_role):
    prefills = [] if missing_role == DisaggregationRole.PREFILL else [_make_runtime("p0", DisaggregationRole.PREFILL)]
    decodes = [] if missing_role == DisaggregationRole.DECODE else [_make_runtime("d0", DisaggregationRole.DECODE)]
    router = _make_router(prefills, decodes)

    with pytest.raises(RuntimeError, match=missing_role.value):
        await router.generate(request_id="req-1", prompt_ids=[1])


@pytest.mark.asyncio
async def test_unhealthy_backend_is_skipped():
    prefills = [
        _make_runtime("p0", DisaggregationRole.PREFILL),
        _make_runtime("p1", DisaggregationRole.PREFILL),
    ]
    decodes = [_make_runtime("d0", DisaggregationRole.DECODE)]
    router = _make_router(prefills, decodes)
    router.mark_unhealthy("p0")

    await asyncio.gather(*[router.generate(request_id=f"req-{index}", prompt_ids=[1]) for index in range(4)])

    assert prefills[0].actor_handle.calls == []
    assert len(prefills[1].actor_handle.calls) == 4


@pytest.mark.asyncio
async def test_register_backends_replaces_pool():
    prefills = [_make_runtime("p0", DisaggregationRole.PREFILL)]
    router = _make_router(prefills, [_make_runtime("d0", DisaggregationRole.DECODE)])
    new_decode = _make_runtime("d1", DisaggregationRole.DECODE)
    router.register_backends(prefills, [new_decode])

    output = await router.generate(request_id="req-1", prompt_ids=[1])
    assert output.extra_fields["served_by"] == "d1"


def test_router_factory_is_sglang_hybrid_specific():
    assert PDRouterFactory.create(PDRouterConfig(backend="ray")).__class__.__name__ == "RayPDRouterController"
    with pytest.raises(NotImplementedError, match="Native sglang_router"):
        PDRouterFactory.create(PDRouterConfig(backend="sglang"))
