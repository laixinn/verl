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
"""GPU-free unit tests for ``verl.workers.rollout.sglang_rollout.pd_topology``.

These tests must not import sglang (directly or transitively) so they stay
lightweight, per the design doc.
"""

from __future__ import annotations

import pytest

from verl.workers.rollout.sglang_rollout.pd_topology import (
    DisaggregationRole,
    HybridPDTopology,
    validate_worker_group_world_size,
)


def test_topology_module_does_not_import_sglang():
    import importlib.util

    spec = importlib.util.find_spec("verl.workers.rollout.sglang_rollout.pd_topology")
    assert spec is not None
    src = spec.loader.get_source(spec.name) if spec.loader else ""
    assert "import sglang" not in src
    assert "from sglang" not in src


def test_1p1d_tp1_maps_ranks_correctly():
    topo = HybridPDTopology.build(tp_size=1, prefill_replicas=1, decode_replicas=1)
    assert topo.world_size == 2

    unit0, tp_local0 = topo.placement_for_global_rank(0)
    assert unit0.role == DisaggregationRole.PREFILL
    assert unit0.role_replica_rank == 0
    assert tp_local0 == 0

    unit1, tp_local1 = topo.placement_for_global_rank(1)
    assert unit1.role == DisaggregationRole.DECODE
    assert unit1.role_replica_rank == 0
    assert tp_local1 == 0


def test_2p6d_tp8_maps_all_64_ranks_exactly_once():
    topo = HybridPDTopology.build(tp_size=8, prefill_replicas=2, decode_replicas=6)
    assert topo.world_size == 64

    seen = set()
    for rank in range(64):
        unit, tp_local_rank = topo.placement_for_global_rank(rank)
        assert 0 <= tp_local_rank < 8
        seen.add(rank)

    assert seen == set(range(64))

    # ranks 0-7 -> prefill 0, ranks 8-15 -> prefill 1
    unit, _ = topo.placement_for_global_rank(0)
    assert unit.role == DisaggregationRole.PREFILL and unit.role_replica_rank == 0
    unit, _ = topo.placement_for_global_rank(8)
    assert unit.role == DisaggregationRole.PREFILL and unit.role_replica_rank == 1
    # ranks 16-23 -> decode 0 ... ranks 56-63 -> decode 5
    unit, _ = topo.placement_for_global_rank(16)
    assert unit.role == DisaggregationRole.DECODE and unit.role_replica_rank == 0
    unit, _ = topo.placement_for_global_rank(63)
    assert unit.role == DisaggregationRole.DECODE and unit.role_replica_rank == 5


@pytest.mark.parametrize(
    "tp_size,prefill_replicas,decode_replicas",
    [(1, 1, 1), (8, 2, 6), (4, 2, 2), (1, 7, 6), (2, 3, 5)],
)
def test_every_unit_has_unique_role_index_actor_name_tuple(tp_size, prefill_replicas, decode_replicas):
    topo = HybridPDTopology.build(tp_size=tp_size, prefill_replicas=prefill_replicas, decode_replicas=decode_replicas)
    triples = {(u.role, u.role_replica_rank, u.primary_actor_name) for u in topo.units}
    assert len(triples) == len(topo.units)
    names = {u.primary_actor_name for u in topo.units}
    assert len(names) == len(topo.units)


@pytest.mark.parametrize(
    "tp_size,prefill_replicas,decode_replicas",
    [(1, 1, 1), (8, 2, 6), (4, 2, 2), (1, 7, 6)],
)
def test_every_global_rank_maps_to_one_unit_and_one_tp_local_rank(tp_size, prefill_replicas, decode_replicas):
    topo = HybridPDTopology.build(tp_size=tp_size, prefill_replicas=prefill_replicas, decode_replicas=decode_replicas)
    seen_pairs = set()
    for rank in range(topo.world_size):
        unit, tp_local_rank = topo.placement_for_global_rank(rank)
        pair = (unit.unit_rank, tp_local_rank)
        assert pair not in seen_pairs
        seen_pairs.add(pair)
        assert unit.global_ranks[tp_local_rank] == rank
    assert len(seen_pairs) == topo.world_size


def test_zero_prefill_or_decode_instances_rejected():
    with pytest.raises(ValueError, match="prefill_replicas"):
        HybridPDTopology.build(tp_size=1, prefill_replicas=0, decode_replicas=1)
    with pytest.raises(ValueError, match="decode_replicas"):
        HybridPDTopology.build(tp_size=1, prefill_replicas=1, decode_replicas=0)


def test_invalid_tp_size_rejected():
    with pytest.raises(ValueError, match="tp_size"):
        HybridPDTopology.build(tp_size=0, prefill_replicas=1, decode_replicas=1)


def test_rank_out_of_range_rejected():
    topo = HybridPDTopology.build(tp_size=2, prefill_replicas=1, decode_replicas=1)
    with pytest.raises(ValueError, match="out of range"):
        topo.placement_for_global_rank(4)
    with pytest.raises(ValueError, match="out of range"):
        topo.placement_for_global_rank(-1)


def test_prefill_and_decode_unit_accessors():
    topo = HybridPDTopology.build(tp_size=2, prefill_replicas=2, decode_replicas=3)
    assert len(topo.prefill_units) == 2
    assert len(topo.decode_units) == 3
    assert all(u.role == DisaggregationRole.PREFILL for u in topo.prefill_units)
    assert all(u.role == DisaggregationRole.DECODE for u in topo.decode_units)


def test_multi_prefill_topology_creates_distinct_bootstrap_reservations():
    """Distinct primary actor names double as distinct bootstrap-reservation owners:
    the launch path reserves one bootstrap socket per prefill unit's primary actor."""
    topo = HybridPDTopology.build(tp_size=4, prefill_replicas=3, decode_replicas=2)
    prefill_actor_names = {u.primary_actor_name for u in topo.prefill_units}
    assert len(prefill_actor_names) == 3


def test_actor_name_pattern_includes_role_and_replica_rank():
    topo = HybridPDTopology.build(tp_size=4, prefill_replicas=2, decode_replicas=2)
    prefill0 = topo.prefill_units[0]
    prefill1 = topo.prefill_units[1]
    decode0 = topo.decode_units[0]
    assert "prefill_0" in prefill0.primary_actor_name
    assert "prefill_1" in prefill1.primary_actor_name
    assert "decode_0" in decode0.primary_actor_name
    assert "node_0" in prefill0.primary_actor_name


def test_actor_name_for_node_multi_node_unit():
    topo = HybridPDTopology.build(tp_size=16, prefill_replicas=1, decode_replicas=1)
    prefill0 = topo.prefill_units[0]
    assert prefill0.actor_name_for_node(0) == prefill0.primary_actor_name
    assert prefill0.actor_name_for_node(1) != prefill0.primary_actor_name
    assert "node_1" in prefill0.actor_name_for_node(1)


def test_deployment_id_namespaces_actor_names():
    topo_a = HybridPDTopology.build(tp_size=1, prefill_replicas=1, decode_replicas=1, deployment_id="a")
    topo_b = HybridPDTopology.build(tp_size=1, prefill_replicas=1, decode_replicas=1, deployment_id="b")
    assert topo_a.units[0].primary_actor_name != topo_b.units[0].primary_actor_name


def test_tp_local_rank_helper_on_unit():
    topo = HybridPDTopology.build(tp_size=4, prefill_replicas=1, decode_replicas=1)
    decode_unit = topo.decode_units[0]
    assert decode_unit.tp_local_rank(decode_unit.global_ranks[2]) == 2


def test_validate_worker_group_world_size_matches():
    topo = HybridPDTopology.build(tp_size=8, prefill_replicas=2, decode_replicas=6)
    validate_worker_group_world_size(topo, 64)  # no raise


def test_validate_worker_group_world_size_mismatch_rejected():
    topo = HybridPDTopology.build(tp_size=8, prefill_replicas=2, decode_replicas=6)
    with pytest.raises(ValueError, match="world_size"):
        validate_worker_group_world_size(topo, 63)
    with pytest.raises(ValueError, match="world_size"):
        validate_worker_group_world_size(topo, 65)


@pytest.mark.parametrize(
    "tp_size,prefill_replicas,decode_replicas",
    [(1, 1, 1), (8, 2, 6), (1, 7, 6), (4, 2, 2)],
)
def test_topology_self_validates_on_build(tp_size, prefill_replicas, decode_replicas):
    # HybridPDTopology.build() calls validate() internally; constructing
    # successfully is itself the assertion that all invariants hold.
    topo = HybridPDTopology.build(tp_size=tp_size, prefill_replicas=prefill_replicas, decode_replicas=decode_replicas)
    topo.validate()
