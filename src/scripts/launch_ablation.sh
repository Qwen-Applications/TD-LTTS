#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/experiment_config.sh"

usage() {
  cat <<'USAGE'
Launch token-routing ablations with the same ECL phase2 reward.

Examples:
  bash src/scripts/launch_ablation.sh --benchmark logicvista --ablation phase2_global --dry-run
  bash src/scripts/launch_ablation.sh --benchmarks mathvista,mmstar --ablation hard_topk --gpus 0,1,2,3
  bash src/scripts/launch_ablation.sh --benchmarks logicvista,mmmu --ablation no_visual_branch --visual-branch aver --tau 0.2 --gpus 0,1,2,3,4,5,6,7 --wait

Ablations:
  phase2_global          ECL reward with global policy loss.
  global_latent_reward   Alias of phase2_global: no token routing.
  soft_visual_entropy    Visual-focus + entropy token router.
  entropy_only           Reasoning router from entropy only.
  visual_only_router     Visual-focus router, reasoning as complement.
  reason_branch_only     Disable visual policy branch.
  visual_branch_only     Disable reasoning policy branch.
  no_visual_branch       Disable visual policy branch under current strict router.
  no_reasoning_branch    Disable reasoning policy branch under current strict router.
  strict_topk            Current strict VPPO image-sensitive top-k router.
  overlap_routing        Image-sensitive visual and entropy reasoning top-k with overlap allowed.
  swapped_routing        Route visual reward to entropy tokens and reasoning reward to image-sensitive tokens.
  hard_topk              Hard top-k visual and reasoning tokens.
  hard_topk_span         Hard top-k tokens constrained by relative spans.
  random_routing         Random hard top-k token routing.
  uniform_routing        Uniformly spaced hard top-k token routing.

Options:
  --benchmarks LIST      Comma-separated benchmark list, or "all".
  --ablation NAME        Required unless ABLATION is set.
  --visual-branch NAME   ecl|aver. Default: aver.
  --tau VALUE            AVER tau when --visual-branch aver. Common: 0.2.
  --mode NAME            sharded|single. Default: sharded.
  --gpus LIST            Passed to launch_sharded.sh.
  --wait                 Wait for sharded runs to finish.
USAGE
  lt_print_usage_common
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark) BENCHMARK="$2"; shift 2 ;;
    --benchmarks) BENCHMARKS="$2"; shift 2 ;;
    --ablation) ABLATION="$2"; shift 2 ;;
    --model-family) MODEL_FAMILY="$2"; shift 2 ;;
    --model) PATH_TO_MODEL="$2"; shift 2 ;;
    --data) PATH_TO_DATA="$2"; shift 2 ;;
    --split) DATASET_SPLIT="$2"; shift 2 ;;
    --output-dir) BASE_OUTPUT_DIR="$2"; shift 2 ;;
    --start) START_DATA_IDX="$2"; shift 2 ;;
    --end) END_DATA_IDX="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --visual-branch) VISUAL_BRANCH_REWARD_MODE="$2"; shift 2 ;;
    --tau|--visual-engagement-tau) VISUAL_ENGAGEMENT_TAU="$2"; shift 2 ;;
    --visual-score-tokens) VISUAL_ENGAGEMENT_MAX_SCORE_TOKENS="$2"; shift 2 ;;
    --wait) WAIT_FOR_COMPLETION=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

MODE="${MODE:-sharded}"
ABLATION="${ABLATION:-}"
VISUAL_BRANCH_REWARD_MODE="${VISUAL_BRANCH_REWARD_MODE:-aver}"
if [[ -z "${ABLATION}" ]]; then
  echo "ABLATION is required. Use --ablation or set ABLATION." >&2
  exit 2
fi

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

  case "${ABLATION}" in
    phase2_global|global_latent_reward)
      export USE_TOKEN_DISENTANGLE=0
      export TOKEN_ROUTER="${TOKEN_ROUTER:-visual_focus_entropy}"
      ;;
    soft_visual_entropy)
      export USE_TOKEN_DISENTANGLE=1
      export TOKEN_ROUTER="${TOKEN_ROUTER:-visual_focus_entropy}"
      ;;
    entropy_only)
      export USE_TOKEN_DISENTANGLE=1
      export TOKEN_ROUTER="${TOKEN_ROUTER:-entropy_only}"
      ;;
    visual_only_router)
      export USE_TOKEN_DISENTANGLE=1
      export TOKEN_ROUTER="${TOKEN_ROUTER:-visual_focus_only}"
      ;;
    reason_branch_only)
      export USE_TOKEN_DISENTANGLE=1
      export TOKEN_ROUTER="${TOKEN_ROUTER:-visual_focus_entropy}"
      export VISUAL_POLICY_WEIGHT=0.0
      export REASON_POLICY_WEIGHT="${REASON_POLICY_WEIGHT:-1.0}"
      ;;
    visual_branch_only)
      export USE_TOKEN_DISENTANGLE=1
      export TOKEN_ROUTER="${TOKEN_ROUTER:-visual_focus_entropy}"
      export VISUAL_POLICY_WEIGHT="${VISUAL_POLICY_WEIGHT:-1.0}"
      export REASON_POLICY_WEIGHT=0.0
      ;;
    no_visual_branch)
      export USE_TOKEN_DISENTANGLE=1
      export TOKEN_ROUTER="${TOKEN_ROUTER:-vppo_strict_topk}"
      export VISUAL_POLICY_WEIGHT=0.0
      export REASON_POLICY_WEIGHT="${REASON_POLICY_WEIGHT:-1.0}"
      ;;
    no_reasoning_branch)
      export USE_TOKEN_DISENTANGLE=1
      export TOKEN_ROUTER="${TOKEN_ROUTER:-vppo_strict_topk}"
      export VISUAL_POLICY_WEIGHT="${VISUAL_POLICY_WEIGHT:-1.0}"
      export REASON_POLICY_WEIGHT=0.0
      ;;
    strict_topk)
      export USE_TOKEN_DISENTANGLE=1
      export TOKEN_ROUTER="${TOKEN_ROUTER:-vppo_strict_topk}"
      ;;
    overlap_routing)
      export USE_TOKEN_DISENTANGLE=1
      export TOKEN_ROUTER="${TOKEN_ROUTER:-vppo_overlap_topk}"
      ;;
    swapped_routing)
      export USE_TOKEN_DISENTANGLE=1
      export TOKEN_ROUTER="${TOKEN_ROUTER:-vppo_swapped_topk}"
      ;;
    hard_topk)
      export USE_TOKEN_DISENTANGLE=1
      export TOKEN_ROUTER="${TOKEN_ROUTER:-hard_topk}"
      ;;
    hard_topk_span)
      export USE_TOKEN_DISENTANGLE=1
      export TOKEN_ROUTER="${TOKEN_ROUTER:-hard_topk_span}"
      ;;
    random_routing)
      export USE_TOKEN_DISENTANGLE=1
      export TOKEN_ROUTER="${TOKEN_ROUTER:-random_topk}"
      ;;
    uniform_routing)
      export USE_TOKEN_DISENTANGLE=1
      export TOKEN_ROUTER="${TOKEN_ROUTER:-uniform_topk}"
      ;;
    *)
      echo "Unknown ABLATION: ${ABLATION}" >&2
      exit 2
      ;;
  esac

  if [[ -n "${BASE_METHOD_TAG}" ]]; then
    export METHOD_TAG="${BASE_METHOD_TAG}"
  else
    export METHOD_TAG="${MODEL_FAMILY}_${BENCHMARK}_${ABLATION}"
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
    if [[ -z "${BASE_METHOD_TAG}" ]]; then
      export METHOD_TAG="${METHOD_TAG}_aver_tau${VISUAL_ENGAGEMENT_TAU}"
    fi
  else
    export VISUAL_BRANCH_REWARD_MODE
  fi

  if [[ -n "${BASE_SUITE_NAME}" ]]; then
    export SUITE_NAME="${BASE_SUITE_NAME}"
  else
    export SUITE_NAME="${MODEL_FAMILY}_${BENCHMARK}_${ABLATION}_${RUN_STAMP}"
  fi

  echo "[ablation] benchmark=${BENCHMARK} model=${MODEL_FAMILY} ablation=${ABLATION} router=${TOKEN_ROUTER} token_disentangle=${USE_TOKEN_DISENTANGLE} visual_branch=${VISUAL_BRANCH_REWARD_MODE} tau=${VISUAL_ENGAGEMENT_TAU:-NA} visual_weight=${VISUAL_POLICY_WEIGHT:-NA} reason_weight=${REASON_POLICY_WEIGHT:-NA}"
  if [[ "${MODE}" == "single" ]]; then
    bash "${SCRIPT_DIR}/run_benchmark.sh" --benchmark "${BENCHMARK}" --model-family "${MODEL_FAMILY}"
  elif [[ "${MODE}" == "sharded" ]]; then
    bash "${SCRIPT_DIR}/launch_sharded.sh" --benchmark "${BENCHMARK}" --model-family "${MODEL_FAMILY}" --suite-name "${SUITE_NAME}"
  else
    echo "Unknown MODE: ${MODE}" >&2
    exit 2
  fi
done
