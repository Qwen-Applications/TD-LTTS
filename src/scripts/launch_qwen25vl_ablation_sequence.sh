#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'USAGE'
Run Qwen2.5-VL-7B ablations sequentially.

Default ablations:
  no_visual_branch,no_reasoning_branch,global_latent_reward,entropy_only,random_routing,uniform_routing

Examples:
  bash src/scripts/launch_qwen25vl_ablation_sequence.sh --gpus 0,1,2,3,4,5,6,7
  bash src/scripts/launch_qwen25vl_ablation_sequence.sh --benchmarks logicvista,mmmu --ablations no_visual_branch,global_latent_reward --dry-run

Options:
  --benchmarks LIST       Comma-separated benchmarks. Default: mathvista,logicvista.
  --ablations LIST        Comma-separated ablations.
  --model-family NAME     Default: qwen25vl7b.
  --gpus LIST             Default: 0,1,2,3,4,5,6,7.
  --visual-branch NAME    ecl|aver. Default: aver.
  --tau VALUE             AVER tau. Default: 0.2.
  --topk-ratio VALUE      Set both visual and reasoning top-k ratios. Default: 0.4.
  --visual-topk-ratio V   Visual top-k ratio. Default: --topk-ratio.
  --reason-topk-ratio V   Reasoning top-k ratio. Default: --topk-ratio.
  --qwen-max-pixels N     Default: 1003520.
  --mode NAME             sharded|single. Default: sharded.
  --no-wait               Do not wait for each sharded benchmark to finish.
  --dry-run               Print commands without running.
USAGE
}

BENCHMARKS="${BENCHMARKS:-mathvista,logicvista}"
ABLATIONS="${ABLATIONS:-no_visual_branch,no_reasoning_branch,global_latent_reward,entropy_only,random_routing,uniform_routing}"
MODEL_FAMILY="${MODEL_FAMILY:-qwen25vl7b}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
VISUAL_BRANCH_REWARD_MODE="${VISUAL_BRANCH_REWARD_MODE:-aver}"
VISUAL_ENGAGEMENT_TAU_VALUE="${VISUAL_ENGAGEMENT_TAU:-0.2}"
TOPK_RATIO="${TOPK_RATIO:-0.4}"
VISUAL_TOPK_RATIO_VALUE="${VISUAL_TOPK_RATIO:-${TOPK_RATIO}}"
REASON_TOPK_RATIO_VALUE="${REASON_TOPK_RATIO:-${TOPK_RATIO}}"
QWEN_MAX_PIXELS_VALUE="${QWEN_MAX_PIXELS:-1003520}"
MODE="${MODE:-sharded}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-1}"
DRY_RUN="${DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmarks) BENCHMARKS="$2"; shift 2 ;;
    --ablations) ABLATIONS="$2"; shift 2 ;;
    --model-family) MODEL_FAMILY="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --visual-branch) VISUAL_BRANCH_REWARD_MODE="$2"; shift 2 ;;
    --tau|--visual-engagement-tau) VISUAL_ENGAGEMENT_TAU_VALUE="$2"; shift 2 ;;
    --topk-ratio)
      TOPK_RATIO="$2"
      VISUAL_TOPK_RATIO_VALUE="$2"
      REASON_TOPK_RATIO_VALUE="$2"
      shift 2
      ;;
    --visual-topk-ratio) VISUAL_TOPK_RATIO_VALUE="$2"; shift 2 ;;
    --reason-topk-ratio) REASON_TOPK_RATIO_VALUE="$2"; shift 2 ;;
    --qwen-max-pixels) QWEN_MAX_PIXELS_VALUE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --no-wait) WAIT_FOR_COMPLETION=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "${PROJECT_ROOT}"
mkdir -p result/launcher_logs

IFS=',' read -r -a ABLATION_ARRAY <<< "${ABLATIONS}"
for ablation in "${ABLATION_ARRAY[@]}"; do
  ablation="$(echo "${ablation}" | xargs)"
  [[ -n "${ablation}" ]] || continue

  unset TOKEN_ROUTER USE_TOKEN_DISENTANGLE VISUAL_POLICY_WEIGHT REASON_POLICY_WEIGHT
  unset METHOD_TAG SUITE_NAME BASE_OUTPUT_DIR LOG_DIR

  export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS_VALUE}"
  export VISUAL_TOPK_RATIO="${VISUAL_TOPK_RATIO_VALUE}"
  export REASON_TOPK_RATIO="${REASON_TOPK_RATIO_VALUE}"
  export VISUAL_BRANCH_REWARD_MODE
  export VISUAL_ENGAGEMENT_TAU="${VISUAL_ENGAGEMENT_TAU_VALUE}"
  export RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)_qwen25vl_${ablation}_${VISUAL_BRANCH_REWARD_MODE}_tau${VISUAL_ENGAGEMENT_TAU_VALUE}_topk${VISUAL_TOPK_RATIO_VALUE}_${REASON_TOPK_RATIO_VALUE}_maxpix${QWEN_MAX_PIXELS_VALUE}"

  cmd=(
    bash "${SCRIPT_DIR}/launch_ablation.sh"
    --benchmarks "${BENCHMARKS}"
    --ablation "${ablation}"
    --model-family "${MODEL_FAMILY}"
    --visual-branch "${VISUAL_BRANCH_REWARD_MODE}"
    --tau "${VISUAL_ENGAGEMENT_TAU_VALUE}"
    --mode "${MODE}"
    --gpus "${GPU_LIST}"
  )
  if [[ "${WAIT_FOR_COMPLETION}" == "1" ]]; then
    cmd+=(--wait)
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    cmd+=(--dry-run)
  fi

  log_path="result/launcher_logs/${RUN_STAMP}.log"
  echo "[sequence] ablation=${ablation} benchmarks=${BENCHMARKS} model=${MODEL_FAMILY} visual_branch=${VISUAL_BRANCH_REWARD_MODE} tau=${VISUAL_ENGAGEMENT_TAU_VALUE} visual_topk=${VISUAL_TOPK_RATIO_VALUE} reason_topk=${REASON_TOPK_RATIO_VALUE}"
  "${cmd[@]}" 2>&1 | tee "${log_path}"
done
