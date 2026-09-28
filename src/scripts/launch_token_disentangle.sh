#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/experiment_config.sh"

usage() {
  cat <<'USAGE'
Launch the current token-disentangled ECL-TTS method.

Examples:
  bash src/scripts/launch_token_disentangle.sh --benchmark mathvista --dry-run
  bash src/scripts/launch_token_disentangle.sh --benchmarks mathvista,mmstar --variant hard_topk --gpus 0,1,2,3
  MODE=single END_DATA_IDX=3 bash src/scripts/launch_token_disentangle.sh --benchmark mathvista

Options:
  --benchmarks LIST      Comma-separated benchmark list, or "all".
  --variant NAME         soft|hard_topk|hard_topk_span|image_sensitivity_topk|vppo_strict_topk|vppo_overlap_topk|vppo_swapped_topk. Default: vppo_strict_topk.
  --token-router NAME    Override token router while keeping the selected method tag.
  --visual-branch NAME   ecl|aver. Default: aver.
  --tau VALUE            AVER tau when --visual-branch aver. Common: 0.2 or 0.05.
  --mode NAME            sharded|single. Default: sharded.
  --gpus LIST            Passed to launch_sharded.sh.
  --wait                 Wait for sharded runs to finish.
USAGE
  lt_print_usage_common
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark) BENCHMARK="$2"; BENCHMARKS="$2"; shift 2 ;;
    --benchmarks) BENCHMARKS="$2"; shift 2 ;;
    --model-family) MODEL_FAMILY="$2"; shift 2 ;;
    --model) PATH_TO_MODEL="$2"; shift 2 ;;
    --data) PATH_TO_DATA="$2"; shift 2 ;;
    --split) DATASET_SPLIT="$2"; shift 2 ;;
    --output-dir) BASE_OUTPUT_DIR="$2"; shift 2 ;;
    --start) START_DATA_IDX="$2"; shift 2 ;;
    --end) END_DATA_IDX="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --token-router) TOKEN_ROUTER="$2"; shift 2 ;;
    --visual-branch) VISUAL_BRANCH_REWARD_MODE="$2"; shift 2 ;;
    --tau|--visual-engagement-tau) VISUAL_ENGAGEMENT_TAU="$2"; shift 2 ;;
    --visual-score-tokens) VISUAL_ENGAGEMENT_MAX_SCORE_TOKENS="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --wait) WAIT_FOR_COMPLETION=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

MODE="${MODE:-sharded}"
VARIANT="${VARIANT:-vppo_strict_topk}"
VISUAL_BRANCH_REWARD_MODE="${VISUAL_BRANCH_REWARD_MODE:-aver}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
BASE_METHOD_TAG="${METHOD_TAG:-}"
BASE_SUITE_NAME="${SUITE_NAME:-}"
BASE_PATH_TO_DATA="${PATH_TO_DATA:-}"
BASE_DATASET_SPLIT="${DATASET_SPLIT:-}"
BASE_MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
BASE_OPTIMIZED_ACCEPTANCE="${OPTIMIZED_ACCEPTANCE:-}"
BASE_ANSWER_FINALIZER_MODE="${ANSWER_FINALIZER_MODE:-}"
BASE_ANSWER_FINALIZER_MAX_NEW_TOKENS="${ANSWER_FINALIZER_MAX_NEW_TOKENS:-}"
export DRY_RUN="${DRY_RUN:-0}"
export WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-0}"
export GPU_LIST="${GPU_LIST:-}"
export START_DATA_IDX="${START_DATA_IDX:-}"
export END_DATA_IDX="${END_DATA_IDX:-}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-}"

if [[ -z "${BENCHMARKS:-}" ]]; then
  BENCHMARKS="${BENCHMARK:-}"
fi
if [[ -z "${BENCHMARKS}" ]]; then
  echo "A benchmark is required. Use --benchmark or --benchmarks." >&2
  exit 2
fi
if [[ "${BENCHMARKS}" == "all" ]]; then
  BENCHMARKS="mathvista,mmstar,logicvista,realworldqa,hallusionbench,scienceqa,mmmu"
fi

lt_set_model_defaults

IFS=',' read -r -a BENCH_ARRAY <<< "${BENCHMARKS}"
for bench in "${BENCH_ARRAY[@]}"; do
  bench="$(echo "${bench}" | xargs)"
  [[ -n "${bench}" ]] || continue

  unset TOTAL_EXAMPLES LT_BENCHMARK_TOTAL
  if [[ -n "${BASE_PATH_TO_DATA}" ]]; then export PATH_TO_DATA="${BASE_PATH_TO_DATA}"; else unset PATH_TO_DATA; fi
  if [[ -n "${BASE_DATASET_SPLIT}" ]]; then export DATASET_SPLIT="${BASE_DATASET_SPLIT}"; else unset DATASET_SPLIT; fi
  if [[ -n "${BASE_MAX_NEW_TOKENS}" ]]; then export MAX_NEW_TOKENS="${BASE_MAX_NEW_TOKENS}"; else unset MAX_NEW_TOKENS; fi
  if [[ -n "${BASE_OPTIMIZED_ACCEPTANCE}" ]]; then export OPTIMIZED_ACCEPTANCE="${BASE_OPTIMIZED_ACCEPTANCE}"; else unset OPTIMIZED_ACCEPTANCE; fi
  if [[ -n "${BASE_ANSWER_FINALIZER_MODE}" ]]; then export ANSWER_FINALIZER_MODE="${BASE_ANSWER_FINALIZER_MODE}"; else unset ANSWER_FINALIZER_MODE; fi
  if [[ -n "${BASE_ANSWER_FINALIZER_MAX_NEW_TOKENS}" ]]; then export ANSWER_FINALIZER_MAX_NEW_TOKENS="${BASE_ANSWER_FINALIZER_MAX_NEW_TOKENS}"; else unset ANSWER_FINALIZER_MAX_NEW_TOKENS; fi
  export BENCHMARK="${bench}"
  lt_set_benchmark_defaults

  if [[ -n "${BASE_METHOD_TAG}" ]]; then
    export METHOD_TAG="${BASE_METHOD_TAG}"
  else
    case "${VARIANT}" in
      soft)
        export TOKEN_ROUTER="${TOKEN_ROUTER:-visual_focus_entropy}"
        export METHOD_TAG="${MODEL_FAMILY}_${BENCHMARK}_td_soft"
        ;;
      hard_topk)
        export TOKEN_ROUTER="${TOKEN_ROUTER:-hard_topk}"
        export METHOD_TAG="${MODEL_FAMILY}_${BENCHMARK}_td_hard_topk"
        ;;
      hard_topk_span)
        export TOKEN_ROUTER="${TOKEN_ROUTER:-hard_topk_span}"
        export METHOD_TAG="${MODEL_FAMILY}_${BENCHMARK}_td_hard_topk_span"
        ;;
      image_sensitivity_topk|vppo_sensitivity_topk|perception_topk)
        export TOKEN_ROUTER="${TOKEN_ROUTER:-image_sensitivity_topk}"
        export METHOD_TAG="${MODEL_FAMILY}_${BENCHMARK}_td_image_sensitivity_topk"
        ;;
      vppo_strict_topk|raw_image_sensitivity_topk)
        export TOKEN_ROUTER="${TOKEN_ROUTER:-vppo_strict_topk}"
        export METHOD_TAG="${MODEL_FAMILY}_${BENCHMARK}_td_vppo_strict_topk"
        ;;
      vppo_overlap_topk|raw_image_sensitivity_overlap_topk)
        export TOKEN_ROUTER="${TOKEN_ROUTER:-vppo_overlap_topk}"
        export METHOD_TAG="${MODEL_FAMILY}_${BENCHMARK}_td_vppo_overlap_topk"
        ;;
      vppo_swapped_topk|raw_image_sensitivity_swapped_topk)
        export TOKEN_ROUTER="${TOKEN_ROUTER:-vppo_swapped_topk}"
        export METHOD_TAG="${MODEL_FAMILY}_${BENCHMARK}_td_vppo_swapped_topk"
        ;;
      *)
        echo "Unknown token-disentangle variant: ${VARIANT}" >&2
        exit 2
        ;;
    esac
  fi

  lt_set_token_disentangle_defaults

  if [[ "${VISUAL_BRANCH_REWARD_MODE}" == "aver" || "${VISUAL_BRANCH_REWARD_MODE}" == "visual_engagement" || "${VISUAL_BRANCH_REWARD_MODE}" == "attention_engagement" ]]; then
    export VISUAL_BRANCH_REWARD_MODE
    export VISUAL_ENGAGEMENT_TAU="${VISUAL_ENGAGEMENT_TAU:-0.2}"
    export VISUAL_ENGAGEMENT_LAYER_INDEX="${VISUAL_ENGAGEMENT_LAYER_INDEX:--1}"
    export VISUAL_ENGAGEMENT_MAX_SCORE_TOKENS="${VISUAL_ENGAGEMENT_MAX_SCORE_TOKENS:-128}"
    export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"
    export VISUAL_ENGAGEMENT_SCORER_ATTN_IMPLEMENTATION="${VISUAL_ENGAGEMENT_SCORER_ATTN_IMPLEMENTATION:-eager}"
    export USE_VISUAL_CLUES=0
    export CLUE_REWARD_WEIGHT=0.0
    export METHOD_TAG="${METHOD_TAG}_aver_tau${VISUAL_ENGAGEMENT_TAU}"
  else
    export VISUAL_BRANCH_REWARD_MODE
  fi

  if [[ -n "${BASE_SUITE_NAME}" ]]; then
    export SUITE_NAME="${BASE_SUITE_NAME}"
  else
    if [[ "${VISUAL_BRANCH_REWARD_MODE}" == "ecl" ]]; then
      export SUITE_NAME="${MODEL_FAMILY}_${BENCHMARK}_${VARIANT}_${RUN_STAMP}"
    else
      export SUITE_NAME="${MODEL_FAMILY}_${BENCHMARK}_${VARIANT}_${VISUAL_BRANCH_REWARD_MODE}_tau${VISUAL_ENGAGEMENT_TAU}_${RUN_STAMP}"
    fi
  fi

  echo "[token-disentangle] benchmark=${BENCHMARK} model=${MODEL_FAMILY} variant=${VARIANT} router=${TOKEN_ROUTER} visual_branch=${VISUAL_BRANCH_REWARD_MODE} tau=${VISUAL_ENGAGEMENT_TAU:-NA}"
  if [[ "${MODE}" == "single" ]]; then
    bash "${SCRIPT_DIR}/run_benchmark.sh" --benchmark "${BENCHMARK}" --model-family "${MODEL_FAMILY}"
  elif [[ "${MODE}" == "sharded" ]]; then
    bash "${SCRIPT_DIR}/launch_sharded.sh" --benchmark "${BENCHMARK}" --model-family "${MODEL_FAMILY}" --suite-name "${SUITE_NAME}"
  else
    echo "Unknown MODE: ${MODE}" >&2
    exit 2
  fi
done
