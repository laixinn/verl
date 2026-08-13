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
"""GPU-free topology helpers for SGLang hybrid Prefill-Decode (PD) disaggregation.

This module computes, once, the mapping from global hybrid-worker rank to a
physical P/D unit (``PDUnitPlacement``) and exposes deterministic actor-name
helpers so that server launch code and ``ServerAdapter`` always agree on
naming. See ``docs/advance/sglang_hybrid_pd_disaggregation.md`` for the full
design.

This module must not import SGLang (or anything that transitively imports it)
so that topology tests stay lightweight and GPU-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DisaggregationRole(str, Enum):
    """Typed PD role, used instead of passing arbitrary strings throughout the code."""

    PREFILL = "prefill"
    DECODE = "decode"


def _actor_name(role: DisaggregationRole, role_replica_rank: int, node_rank: int, deployment_id: str = "") -> str:
    """Deterministic actor name shared by launch code and ``ServerAdapter``.

    Pattern: ``sglang_pd_hybrid_<role>_<role_replica_rank>_node_<node_rank>``,
    optionally prefixed by a deployment identifier so multiple concurrent
    deployments (not yet supported end-to-end) don't collide.
    """
    prefix = f"sglang_pd_hybrid{f'_{deployment_id}' if deployment_id else ''}"
    return f"{prefix}_{role.value}_{role_replica_rank}_node_{node_rank}"


@dataclass(frozen=True)
class PDUnitPlacement:
    """Placement of one physical P/D unit (a ``tp``-sized SGLang instance).

    Attributes:
        unit_rank: Global unit index, unique across prefill and decode units
            combined. Canonical ordering is all prefill units followed by all
            decode units (see ``HybridPDTopology``).
        role: Whether this unit is a prefill or decode replica.
        role_replica_rank: Index of this unit within its role's pool
            (0-indexed), used for user-facing metrics and naming.
        global_ranks: The tuple of global hybrid-worker ranks assigned to this
            unit, in TP-local-rank order.
        tp_size: Tensor-parallel size of this unit (equal for every P/D unit).
        primary_actor_name: Deterministic actor name of the node-rank-0 server
            actor for this unit; shared by launch code and ``ServerAdapter``.
    """

    unit_rank: int
    role: DisaggregationRole
    role_replica_rank: int
    global_ranks: tuple[int, ...]
    tp_size: int
    primary_actor_name: str
    deployment_id: str = ""

    @property
    def begin_rank(self) -> int:
        return self.global_ranks[0]

    def actor_name_for_node(self, node_rank: int) -> str:
        """Actor name for a given node rank within this unit (multi-node units)."""
        return _actor_name(self.role, self.role_replica_rank, node_rank, self.deployment_id)

    def tp_local_rank(self, global_rank: int) -> int:
        """TP-local rank of ``global_rank`` within this unit."""
        return self.global_ranks.index(global_rank)


@dataclass(frozen=True)
class HybridPDTopology:
    """Complete rank-to-physical-unit mapping for one hybrid PD deployment.

    The canonical ordering is all prefill units followed by all decode units:

    ```
    unit_rank = global_rank // tp_size
    tp_local_rank = global_rank % tp_size

    if unit_rank < prefill_replicas:
        role = PREFILL, role_replica_rank = unit_rank
    else:
        role = DECODE, role_replica_rank = unit_rank - prefill_replicas
    ```
    """

    tp_size: int
    prefill_replicas: int
    decode_replicas: int
    units: tuple[PDUnitPlacement, ...]
    deployment_id: str = ""

    @property
    def world_size(self) -> int:
        return (self.prefill_replicas + self.decode_replicas) * self.tp_size

    @property
    def prefill_units(self) -> tuple[PDUnitPlacement, ...]:
        return self.units[: self.prefill_replicas]

    @property
    def decode_units(self) -> tuple[PDUnitPlacement, ...]:
        return self.units[self.prefill_replicas :]

    def unit_for_rank(self, rank: int) -> PDUnitPlacement:
        """Return the physical unit that owns a given global hybrid-worker rank."""
        if not (0 <= rank < self.world_size):
            raise ValueError(f"rank {rank} out of range for world_size={self.world_size}")
        unit_rank = rank // self.tp_size
        return self.units[unit_rank]

    def placement_for_global_rank(self, rank: int) -> tuple[PDUnitPlacement, int]:
        """Return the physical unit and TP-local rank for a worker rank."""
        unit = self.unit_for_rank(rank)
        tp_local_rank = rank % self.tp_size
        return unit, tp_local_rank

    @classmethod
    def build(
        cls,
        tp_size: int,
        prefill_replicas: int,
        decode_replicas: int,
        deployment_id: str = "",
    ) -> HybridPDTopology:
        """Build and validate the topology for the given P/D counts and TP size.

        Every P/D unit is modeled as single-node here (``node_rank=0`` in its
        primary actor name); multi-node units reuse this same numbering with
        additional per-node actors created by the launch path (node_rank > 0
        never owns the HTTP/control-plane primary).
        """
        if tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {tp_size}")
        if prefill_replicas < 1:
            raise ValueError(f"prefill_replicas must be >= 1, got {prefill_replicas}")
        if decode_replicas < 1:
            raise ValueError(f"decode_replicas must be >= 1, got {decode_replicas}")

        units: list[PDUnitPlacement] = []
        total_units = prefill_replicas + decode_replicas
        for unit_rank in range(total_units):
            if unit_rank < prefill_replicas:
                role = DisaggregationRole.PREFILL
                role_replica_rank = unit_rank
            else:
                role = DisaggregationRole.DECODE
                role_replica_rank = unit_rank - prefill_replicas

            begin = unit_rank * tp_size
            global_ranks = tuple(range(begin, begin + tp_size))
            primary_actor_name = _actor_name(role, role_replica_rank, node_rank=0, deployment_id=deployment_id)
            units.append(
                PDUnitPlacement(
                    unit_rank=unit_rank,
                    role=role,
                    role_replica_rank=role_replica_rank,
                    global_ranks=global_ranks,
                    tp_size=tp_size,
                    primary_actor_name=primary_actor_name,
                    deployment_id=deployment_id,
                )
            )

        topology = cls(
            tp_size=tp_size,
            prefill_replicas=prefill_replicas,
            decode_replicas=decode_replicas,
            units=tuple(units),
            deployment_id=deployment_id,
        )
        topology.validate()
        return topology

    def validate(self) -> None:
        """Sanity-check invariants: exact rank coverage, unique names, unique triples."""
        expected_world_size = self.world_size
        seen_ranks: set[int] = set()
        seen_names: set[str] = set()
        seen_triples: set[tuple[DisaggregationRole, int]] = set()

        for unit in self.units:
            if len(unit.global_ranks) != self.tp_size:
                raise ValueError(
                    f"unit_rank={unit.unit_rank} has {len(unit.global_ranks)} ranks, expected tp_size={self.tp_size}"
                )
            for rank in unit.global_ranks:
                if rank in seen_ranks:
                    raise ValueError(f"global rank {rank} assigned to more than one PD unit")
                seen_ranks.add(rank)

            if unit.primary_actor_name in seen_names:
                raise ValueError(f"duplicate actor name {unit.primary_actor_name!r} across PD units")
            seen_names.add(unit.primary_actor_name)

            triple = (unit.role, unit.role_replica_rank)
            if triple in seen_triples:
                raise ValueError(f"duplicate (role, role_replica_rank) {triple} across PD units")
            seen_triples.add(triple)

        if len(seen_ranks) != expected_world_size:
            raise ValueError(
                f"PD topology covers {len(seen_ranks)} ranks, expected exactly {expected_world_size} "
                f"(prefill_replicas={self.prefill_replicas}, decode_replicas={self.decode_replicas}, "
                f"tp_size={self.tp_size})"
            )
        if seen_ranks != set(range(expected_world_size)):
            raise ValueError("PD topology does not cover global ranks [0, world_size) contiguously and exactly once")


def validate_worker_group_world_size(topology: HybridPDTopology, worker_group_world_size: int) -> None:
    """Fail closed unless the hybrid worker group has exactly the PD topology's world size.

    Requiring an exact match prevents silently unused trainer GPUs and keeps
    rank-to-instance mapping deterministic (see design doc "Required invariants").
    """
    if worker_group_world_size != topology.world_size:
        raise ValueError(
            f"worker_group.world_size ({worker_group_world_size}) must equal "
            f"(prefill_replicas + decode_replicas) * tp = "
            f"({topology.prefill_replicas} + {topology.decode_replicas}) * {topology.tp_size} "
            f"= {topology.world_size}"
        )
