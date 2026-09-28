#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SRC_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/lib/experiment_config.sh"

usage() {
  cat <<'USAGE'
Launch a benchmark over N GPU shards. The number of shards defaults to the
number of comma-separated GPU ids in GPU_LIST.

Examples:
  bash src/scripts/launch_sharded.sh --benchmark mathvista --gpus 0,1,2,3 --dry-run
  BENCHMARK=mmstar GPU_LIST=0,1,2,3,4,5,6,7 WAIT_FOR_COMPLETION=1 bash src/scripts/launch_sharded.sh

Extra options:
  --gpus LIST            Comma-separated GPU ids, or semicolon-separated GPU groups
                         such as "0,1;2,3;4,5;6,7". Default: 0,1,2,3,4,5,6,7
  --num-shards N         Use first N GPUs from GPU_LIST. Default: all GPUs.
  --suite-name NAME      Result suite name.
  --total N              Total examples used for sharding.
  --wait                 Wait for all shards to finish.
USAGE
  lt_print_usage_common
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark) BENCHMARK="$2"; shift 2 ;;
    --model-family) MODEL_FAMILY="$2"; shift 2 ;;
    --model) PATH_TO_MODEL="$2"; shift 2 ;;
    --data) PATH_TO_DATA="$2"; shift 2 ;;
    --split) DATASET_SPLIT="$2"; shift 2 ;;
    --output-dir) BASE_OUTPUT_DIR="$2"; shift 2 ;;
    --start) START_DATA_IDX="$2"; shift 2 ;;
    --end) END_DATA_IDX="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --num-shards) NUM_SHARDS="$2"; shift 2 ;;
    --suite-name) SUITE_NAME="$2"; shift 2 ;;
    --total) TOTAL_EXAMPLES="$2"; shift 2 ;;
    --wait) WAIT_FOR_COMPLETION=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "${PROJECT_ROOT}"
lt_set_model_defaults
lt_set_benchmark_defaults

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
if [[ "${GPU_LIST}" == *";"* ]]; then
  IFS=';' read -r -a GPUS <<< "${GPU_LIST}"
else
  IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
fi
NUM_SHARDS="${NUM_SHARDS:-${#GPUS[@]}}"
if (( NUM_SHARDS <= 0 )); then
  echo "NUM_SHARDS must be positive." >&2
  exit 2
fi
if (( NUM_SHARDS > ${#GPUS[@]} )); then
  echo "NUM_SHARDS (${NUM_SHARDS}) cannot exceed GPU count (${#GPUS[@]})." >&2
  exit 2
fi

RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/result}"
METHOD_TAG="${METHOD_TAG:-run}"
SUITE_NAME="${SUITE_NAME:-${MODEL_FAMILY}_${BENCHMARK}_${METHOD_TAG}_$(date -u +%Y%m%d_%H%M%S)}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${RESULT_ROOT}/suites/${SUITE_NAME}/${BENCHMARK}/${METHOD_TAG}}"
LOG_DIR="${LOG_DIR:-${BASE_OUTPUT_DIR}/logs}"
START_DATA_IDX="${START_DATA_IDX:-0}"
END_DATA_IDX="${END_DATA_IDX:-${TOTAL_EXAMPLES}}"
WAIT_READY="${WAIT_READY:-1}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
LAUNCH_DELAY="${LAUNCH_DELAY:-3}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-0}"
DRY_RUN="${DRY_RUN:-0}"
CONDA_ENV="${CONDA_ENV:-}"
CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"

if (( END_DATA_IDX < 0 )); then
  END_DATA_IDX="${TOTAL_EXAMPLES}"
fi
if (( END_DATA_IDX < START_DATA_IDX )); then
  echo "END_DATA_IDX must be >= START_DATA_IDX." >&2
  exit 2
fi

JOB_FILE="${LOG_DIR}/${BENCHMARK}_${METHOD_TAG}_jobs.txt"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${LOG_DIR}" "${BASE_OUTPUT_DIR}"
  : > "${JOB_FILE}"
fi

if command -v rg >/dev/null 2>&1; then
  LOG_SEARCH_CMD=(rg -q)
else
  LOG_SEARCH_CMD=(grep -q)
fi

declare -a PIDS=()
declare -a TAGS=()
declare -a LOG_PATHS=()

range_len=$(( END_DATA_IDX - START_DATA_IDX ))
for (( idx = 0; idx < NUM_SHARDS; idx++ )); do
  gpu="${GPUS[$idx]}"
  start_idx=$(( START_DATA_IDX + (range_len * idx) / NUM_SHARDS ))
  end_idx=$(( START_DATA_IDX + (range_len * (idx + 1)) / NUM_SHARDS ))
  shard_name="g${idx}_${start_idx}_${end_idx}"
  shard_output_dir="${BASE_OUTPUT_DIR}/g${idx}"
  log_path="${LOG_DIR}/${BENCHMARK}_${METHOD_TAG}_${shard_name}.log"
  tag="${BENCHMARK}_${METHOD_TAG}_${shard_name}"

  run_command=(bash "${SCRIPT_DIR}/run_benchmark.sh" --benchmark "${BENCHMARK}" --model-family "${MODEL_FAMILY}" --output-dir "${shard_output_dir}" --start "${start_idx}" --end "${end_idx}")
  if [[ -n "${CONDA_ENV}" ]]; then
    run_command=(bash -lc "export PATH=\"${CONDA_BASE}/envs/${CONDA_ENV}/bin:\$PATH\"; export CONDA_PREFIX=\"${CONDA_BASE}/envs/${CONDA_ENV}\"; export CONDA_DEFAULT_ENV=\"${CONDA_ENV}\"; export CONDA_SHLVL=1; bash \"${SCRIPT_DIR}/run_benchmark.sh\" --benchmark \"${BENCHMARK}\" --model-family \"${MODEL_FAMILY}\" --output-dir \"${shard_output_dir}\" --start \"${start_idx}\" --end \"${end_idx}\"")
  fi

  echo "Launching ${tag} on GPU ${gpu}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN CUDA_VISIBLE_DEVICES=${gpu} OUTPUT_DIR=${shard_output_dir} START_DATA_IDX=${start_idx} END_DATA_IDX=${end_idx}"
    lt_print_command "${run_command[@]}"
    continue
  fi

  mkdir -p "${shard_output_dir}"
  env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}" \
    nohup "${run_command[@]}" > "${log_path}" 2>&1 &
  pid=$!

  PIDS+=("${pid}")
  TAGS+=("${tag}")
  LOG_PATHS+=("${log_path}")
  echo "${pid} ${gpu} ${tag} ${shard_output_dir}" | tee -a "${JOB_FILE}"

  if [[ "${WAIT_READY}" == "1" ]]; then
    waited=0
    while kill -0 "${pid}" 2>/dev/null; do
      if [[ -f "${log_path}" ]] && "${LOG_SEARCH_CMD[@]}" "Start to evaluate" "${log_path}"; then
        echo "READY ${tag}"
        break
      fi
      sleep "${POLL_INTERVAL}"
      waited=$(( waited + POLL_INTERVAL ))
      if (( waited >= READY_TIMEOUT )); then
        echo "WARN timeout waiting for ${tag} to become ready" >&2
        break
      fi
    done
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ERROR ${tag} exited before becoming ready; check ${log_path}" >&2
      tail -n 40 "${log_path}" >&2 || true
      exit 1
    fi
  fi

  sleep "${LAUNCH_DELAY}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

echo "Launched ${BENCHMARK}/${METHOD_TAG} with ${NUM_SHARDS} shard(s)."
echo "Logs: ${LOG_DIR}"

if [[ "${WAIT_FOR_COMPLETION}" == "1" ]]; then
  fail_count=0
  for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    tag="${TAGS[$idx]}"
    log_path="${LOG_PATHS[$idx]}"
    if wait "${pid}"; then
      echo "DONE ${tag}"
    else
      echo "FAIL ${tag}; check ${log_path}" >&2
      tail -n 40 "${log_path}" >&2 || true
      fail_count=$(( fail_count + 1 ))
    fi
  done
  if (( fail_count > 0 )); then
    echo "${BENCHMARK}/${METHOD_TAG} finished with ${fail_count} failing shard(s)." >&2
    exit 1
  fi

  summary_path="${BASE_OUTPUT_DIR}/aggregate_summary.json"
  if ! python "${SCRIPT_DIR}/summarize_sharded_results.py" \
    --base-output-dir "${BASE_OUTPUT_DIR}" \
    --benchmark "${BENCHMARK}" \
    --method-tag "${METHOD_TAG}" \
    --expected-shards "${NUM_SHARDS}" \
    --expected-total "${range_len}" \
    --write-json "${summary_path}"; then
    echo "WARN failed to summarize sharded results under ${BASE_OUTPUT_DIR}" >&2
  fi
fi
