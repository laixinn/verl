#!/usr/bin/env bash
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

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
HARNESS="${SCRIPT_DIR}/run_function_reward.sh"

NUM_GPUS=${NUM_GPUS:-8}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
TRAIN_FILES=${TRAIN_FILES:-${HOME}/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-${HOME}/data/gsm8k/test.parquet}
PD_IB_DEVICE=${PD_IB_DEVICE:-}

if [[ "${NUM_GPUS}" -ne 8 ]]; then
    echo "SGLang hybrid-PD PPO E2E requires NUM_GPUS=8; got ${NUM_GPUS}." >&2
    exit 1
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Model config not found at ${MODEL_PATH}/config.json." >&2
    exit 1
fi
for data_file in "${TRAIN_FILES}" "${VAL_FILES}"; do
    if [[ ! -f "${data_file}" ]]; then
        echo "GSM8K parquet file not found: ${data_file}" >&2
        exit 1
    fi
done

python3 - <<'PY'
import torch

import ray  # noqa: F401
import sglang  # noqa: F401
from mooncake.engine import TransferEngine  # noqa: F401

available = torch.cuda.device_count()
if available < 8:
    raise SystemExit(f"SGLang hybrid-PD PPO E2E requires at least 8 CUDA GPUs; found {available}.")
PY

RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/verl-sglang-hybrid-pd.XXXXXX")"
LOG_FILE="${RUN_DIR}/hybrid_pd_ppo.log"

cleanup() {
    status=$?
    trap - EXIT
    if [[ "${status}" -eq 0 ]]; then
        rm -rf "${RUN_DIR}"
    else
        echo "Hybrid-PD PPO logs retained at ${RUN_DIR}." >&2
    fi
    exit "${status}"
}
trap cleanup EXIT

ib_device_override="actor_rollout_ref.rollout.disaggregation.ib_device=null"
if [[ -n "${PD_IB_DEVICE}" ]]; then
    ib_device_override="actor_rollout_ref.rollout.disaggregation.ib_device='${PD_IB_DEVICE}'"
fi

cd "${RUN_DIR}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
NUM_GPUS="${NUM_GPUS}" \
MODEL_ID="${MODEL_ID}" \
MODEL_PATH="${MODEL_PATH}" \
TRAIN_FILES="${TRAIN_FILES}" \
VAL_FILES="${VAL_FILES}" \
MAX_PROMPT_LEN=256 \
MAX_RESPONSE_LEN=64 \
TOTAL_TRAIN_STEPS=2 \
ENGINE=sglang \
STRATEGY=fsdp \
ADV_ESTIMATOR=gae \
LOAD_FORMAT=dummy \
GPU_MEMORY_UTILIZATION=0.6 \
ENABLE_CHUNKED_PREFILL=False \
VAL_BEFORE_TRAIN=False \
TEST_FREQ=-1 \
SAVE_FREQ=-1 \
RESUME_MODE=disable \
VERL_EXP_NAME=sglang-hybrid-pd-ppo-regression \
bash "${HARNESS}" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_model_len=384 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=naive \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512 \
    actor_rollout_ref.rollout.disaggregation.enabled=True \
    actor_rollout_ref.rollout.disaggregation.prefill_replicas=2 \
    actor_rollout_ref.rollout.disaggregation.decode_replicas=2 \
    actor_rollout_ref.rollout.disaggregation.decode_tensor_model_parallel_size=null \
    actor_rollout_ref.rollout.disaggregation.transfer_backend=mooncake \
    actor_rollout_ref.rollout.disaggregation.bootstrap_port=null \
    "${ib_device_override}" \
    ++actor_rollout_ref.rollout.disaggregation.router.backend=ray \
    ++actor_rollout_ref.rollout.disaggregation.router.prefill_policy=least_inflight \
    ++actor_rollout_ref.rollout.disaggregation.router.decode_policy=least_inflight \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=sync \
    2>&1 | tee "${LOG_FILE}"

python3 - "${LOG_FILE}" <<'PY'
import math
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
text = log_path.read_text(errors="replace")
markers = [
    "LLMServerManager: ['sglang_pd_router']",
    "SGLangPDReplica prefill unit_rank=0 role_replica_rank=0 launched:",
    "SGLangPDReplica prefill unit_rank=1 role_replica_rank=1 launched:",
    "SGLangPDReplica decode unit_rank=2 role_replica_rank=0 launched:",
    "SGLangPDReplica decode unit_rank=3 role_replica_rank=1 launched:",
]
missing = [marker for marker in markers if marker not in text]
if missing:
    raise SystemExit("Missing hybrid-PD PPO regression markers: " + ", ".join(missing))

required_metrics = [
    "actor/pg_loss",
    "timing_s/gen",
    "timing_s/update_critic",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "training/off_policy/trajectory_spans/mean",
    "training/off_policy/trajectory_staleness/mean",
    "training/off_policy/trajectory_staleness_worst/mean",
]
metrics_line = next(
    (
        line
        for line in text.splitlines()
        if "step:2" in line and all(f"{key}:" in line for key in required_metrics)
    ),
    None,
)
if metrics_line is None:
    raise SystemExit(
        "No step-2 metrics line contained generation, actor/critic update, "
        "policy loss, weight-sync, and model-version metrics."
    )


def metric_value(key: str) -> float:
    raw = metrics_line.split(f"{key}:", 1)[1].split(" - ", 1)[0]
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", raw)
    if not numbers:
        raise SystemExit(f"Could not parse {key} from step-2 metrics: {raw}")
    value = float(numbers[-1])
    if not math.isfinite(value):
        raise SystemExit(f"Non-finite {key} in step-2 metrics: {raw}")
    return value


for key in required_metrics:
    metric_value(key)
if metric_value("training/off_policy/trajectory_spans/mean") != 1.0:
    raise SystemExit("Step-2 trajectories crossed more than one rollout model version.")
if metric_value("training/off_policy/trajectory_staleness/mean") != 0.0:
    raise SystemExit("Step-2 rollout did not use the step-1 synchronized weights.")
if metric_value("training/off_policy/trajectory_staleness_worst/mean") != 0.0:
    raise SystemExit("Some step-2 trajectories used stale rollout weights.")
PY

echo "SGLang hybrid-PD PPO E2E passed."
