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
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig

__all__ = ["DisaggregationConfig", "PDRouterConfig"]

_ALLOWED_BACKENDS = ("nixl", "mooncake", "ascend", "mori", "fake")
_ALLOWED_ROUTER_POLICIES = ("least_inflight", "random")


@dataclass
class PDRouterConfig(BaseConfig):
    """SGLang hybrid-PD router settings."""

    backend: str = "ray"
    prefill_policy: str = "least_inflight"
    decode_policy: str = "least_inflight"

    def __post_init__(self) -> None:
        if self.backend not in ("ray", "sglang"):
            raise ValueError(f"disaggregation.router.backend={self.backend!r} must be 'ray' or 'sglang'")
        for role, policy in (("prefill", self.prefill_policy), ("decode", self.decode_policy)):
            if policy not in _ALLOWED_ROUTER_POLICIES:
                raise ValueError(
                    f"disaggregation.router.{role}_policy={policy!r} not in {_ALLOWED_ROUTER_POLICIES}"
                )


_ALLOWED_MOONCAKE_PROTOCOLS = ("nvlink", "local", "rdma", "tcp")


@dataclass
class DisaggregationConfig(BaseConfig):
    """Prefill-Decode disaggregation knobs."""

    enabled: bool = False
    prefill_replicas: int = 1
    decode_replicas: int = 1
    decode_tensor_model_parallel_size: Optional[int] = None
    transfer_backend: str = "nixl"
    bootstrap_port: Optional[int] = None
    ib_device: Optional[str] = None
    mooncake_protocol: str = "nvlink"
    router: PDRouterConfig = field(default_factory=PDRouterConfig)

    def __post_init__(self) -> None:
        if isinstance(self.router, dict):
            object.__setattr__(self, "router", PDRouterConfig(**self.router))
        if not self.enabled:
            return
        if self.transfer_backend not in _ALLOWED_BACKENDS:
            raise ValueError(f"disaggregation.transfer_backend={self.transfer_backend!r} not in {_ALLOWED_BACKENDS}")
        if self.prefill_replicas < 1 or self.decode_replicas < 1:
            raise ValueError(
                f"disaggregation requires >=1 prefill and >=1 decode replica "
                f"(got prefill_replicas={self.prefill_replicas}, decode_replicas={self.decode_replicas})"
            )
        if self.bootstrap_port is not None and not (0 < self.bootstrap_port < 65536):
            raise ValueError(f"bootstrap_port out of range: {self.bootstrap_port}")
        if self.bootstrap_port is not None and self.prefill_replicas > 1:
            raise ValueError(
                "disaggregation.bootstrap_port can only be set when prefill_replicas == 1; "
                f"got prefill_replicas={self.prefill_replicas}. With multiple prefill instances each "
                "must reserve its own unique bootstrap port automatically -- do not derive them as "
                "bootstrap_port + index, which can collide with unrelated processes."
            )
        if self.transfer_backend == "mooncake" and self.mooncake_protocol not in _ALLOWED_MOONCAKE_PROTOCOLS:
            raise ValueError(
                f"disaggregation.mooncake_protocol={self.mooncake_protocol!r} not in {_ALLOWED_MOONCAKE_PROTOCOLS}"
            )

    def effective_decode_tp(self, prefill_tp: int) -> int:
        """Resolve decode TP (defaults to ``prefill_tp``). Test-only helper; runtime paths
        must inline this because OmegaConf/Ray serialization drops dataclass methods."""
        if self.decode_tensor_model_parallel_size is not None:
            return self.decode_tensor_model_parallel_size
        return prefill_tp

    def validate_equal_pd_tp(self, prefill_tp: int) -> None:
        """Fail closed unless decode TP is unset or explicitly equal to prefill TP.

        The hybrid PD replica set requires equal prefill/decode TP (see design doc);
        ``decode_tensor_model_parallel_size`` is retained only for backward
        compatibility and must be either ``None`` or equal to ``prefill_tp``.
        """
        if not self.enabled:
            return
        if self.decode_tensor_model_parallel_size is not None and self.decode_tensor_model_parallel_size != prefill_tp:
            raise ValueError(
                "The hybrid PD implementation requires equal prefill/decode tensor parallel size. "
                f"rollout.disaggregation.decode_tensor_model_parallel_size="
                f"{self.decode_tensor_model_parallel_size} must be None or equal to "
                f"rollout.tensor_model_parallel_size={prefill_tp}. Asymmetric prefill/decode TP is "
                "explicitly out of scope for this implementation."
            )
