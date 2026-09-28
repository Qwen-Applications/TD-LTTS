#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SRC_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/lib/experiment_config.sh"

usage() {
  cat <<'USAGE'
Run one benchmark shard or a full benchmark in a single process.

Examples:
  bash src/scripts/run_benchmark.sh --benchmark mathvista --end 3 --dry-run
  BENCHMARK=mmstar START_DATA_IDX=0 END_DATA_IDX=10 bash src/scripts/run_benchmark.sh
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
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --start) START_DATA_IDX="$2"; shift 2 ;;
    --end) END_DATA_IDX="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "${PROJECT_ROOT}"

if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT
fi
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

lt_set_model_defaults
lt_set_benchmark_defaults

RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/result}"
METHOD_TAG="${METHOD_TAG:-run}"
RUN_NAME="${RUN_NAME:-${BENCHMARK}_${METHOD_TAG}_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULT_ROOT}/${BENCHMARK}/${RUN_NAME}}"
START_DATA_IDX="${START_DATA_IDX:-0}"
END_DATA_IDX="${END_DATA_IDX:--1}"
DEVICE="${DEVICE:-cuda}"
PROCESSOR_USE_FAST="${PROCESSOR_USE_FAST:-0}"
QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-}"
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-}"
MODEL_DEVICE_MAP="${MODEL_DEVICE_MAP:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"
ALLOW_CUDNN_SDPA="${ALLOW_CUDNN_SDPA:-0}"
MAX_NUM_STEPS="${MAX_NUM_STEPS:-4}"
SAVE_EVERY="${SAVE_EVERY:-10}"
RESUME="${RESUME:-0}"
REWARD_THRESHOLD="${REWARD_THRESHOLD:--0.2}"
K="${K:-${rho:-0.5}}"
LR="${LR:-${lr:-0.05}}"
SOLVER_PROMPT_IDX="${SOLVER_PROMPT_IDX:-${solver_prompt_idx:-1}}"
LATENT_START_STRATEGY="${LATENT_START_STRATEGY:-full}"
ANSWER_TAIL_TOKENS="${ANSWER_TAIL_TOKENS:-0}"
ANSWER_START_FALLBACK_RATIO="${ANSWER_START_FALLBACK_RATIO:-0.8}"
LATENT_ANCHOR_WEIGHT="${LATENT_ANCHOR_WEIGHT:-0.05}"
LOGIT_LR="${LOGIT_LR:-0.01}"
NOISE_STD="${NOISE_STD:-0.3}"
NOISE_CANDIDATES="${NOISE_CANDIDATES:-3}"
CONFIDENCE_REWARD_MIX="${CONFIDENCE_REWARD_MIX:-0.5}"
BASELINE_MODE="${BASELINE_MODE:-none}"
BASELINE_NUM_CANDIDATES="${BASELINE_NUM_CANDIDATES:-4}"
BASELINE_TEMPERATURE="${BASELINE_TEMPERATURE:-0.7}"
BASELINE_TOP_P="${BASELINE_TOP_P:-0.95}"
ORIGINAL_LATENTSEEK_RULE_MODE="${ORIGINAL_LATENTSEEK_RULE_MODE:-text}"
ORIGINAL_LATENTSEEK_VERIFIER_MAX_NEW_TOKENS="${ORIGINAL_LATENTSEEK_VERIFIER_MAX_NEW_TOKENS:-4096}"
ORIGINAL_LATENTSEEK_USE_COMPACT_VERIFIERS="${ORIGINAL_LATENTSEEK_USE_COMPACT_VERIFIERS:-0}"
VISUAL_REWARD_MODE="${VISUAL_REWARD_MODE:-likelihood}"
VISUAL_PRIOR_WEIGHT="${VISUAL_PRIOR_WEIGHT:-1.0}"
CONTRASTIVE_TEMPERATURE="${CONTRASTIVE_TEMPERATURE:-1.0}"
USE_EVIDENCE_CALIBRATION="${USE_EVIDENCE_CALIBRATION:-0}"
ANSWER_REWARD_WEIGHT="${ANSWER_REWARD_WEIGHT:-1.0}"
MARGIN_REWARD_WEIGHT="${MARGIN_REWARD_WEIGHT:-0.5}"
USE_VISUAL_CLUES="${USE_VISUAL_CLUES:-0}"
NUM_VISUAL_CLUES="${NUM_VISUAL_CLUES:-3}"
CLUE_REWARD_WEIGHT="${CLUE_REWARD_WEIGHT:-0.3}"
CLUE_MAX_NEW_TOKENS="${CLUE_MAX_NEW_TOKENS:-96}"
USE_CROP_EVIDENCE="${USE_CROP_EVIDENCE:-0}"
CROP_REWARD_WEIGHT="${CROP_REWARD_WEIGHT:-0.4}"
TOP_K_POSITIVE_CROPS="${TOP_K_POSITIVE_CROPS:-2}"
MAX_NEGATIVE_VIEWS="${MAX_NEGATIVE_VIEWS:-3}"
CROP_GRID_SIZES="${CROP_GRID_SIZES:-2,3}"
MAX_CROP_CANDIDATES_PER_STEP="${MAX_CROP_CANDIDATES_PER_STEP:-2}"
CANDIDATE_AGGREGATION="${CANDIDATE_AGGREGATION:-majority}"
AGGREGATION_TEMPERATURE="${AGGREGATION_TEMPERATURE:-0.5}"
DEBUG_VISUAL_FOCUS="${DEBUG_VISUAL_FOCUS:-0}"
DEBUG_LOSS_TERMS="${DEBUG_LOSS_TERMS:-0}"
DEBUG_REWARD_TERMS="${DEBUG_REWARD_TERMS:-0}"
DEBUG_CANDIDATE_REWARDS="${DEBUG_CANDIDATE_REWARDS:-0}"
DEBUG_CANDIDATE_CONDITIONED_CLUES="${DEBUG_CANDIDATE_CONDITIONED_CLUES:-0}"
CANDIDATE_CONDITIONED_CLUE_TOP_K="${CANDIDATE_CONDITIONED_CLUE_TOP_K:-2}"
CANDIDATE_CONDITIONED_CLUE_MAX_NEW_TOKENS="${CANDIDATE_CONDITIONED_CLUE_MAX_NEW_TOKENS:-64}"
USE_TOKEN_DISENTANGLE="${USE_TOKEN_DISENTANGLE:-0}"
TOKEN_ROUTER="${TOKEN_ROUTER:-vppo_strict_topk}"
REASON_ROUTER_SIGNAL="${REASON_ROUTER_SIGNAL:-entropy}"
IMAGE_SENSITIVITY_DROP_PROB="${IMAGE_SENSITIVITY_DROP_PROB:-0.5}"
IMAGE_SENSITIVITY_PATCH_SIZE="${IMAGE_SENSITIVITY_PATCH_SIZE:-14}"
VISUAL_ROUTER_THRESHOLD="${VISUAL_ROUTER_THRESHOLD:-0.5}"
ENTROPY_ROUTER_THRESHOLD="${ENTROPY_ROUTER_THRESHOLD:-0.0}"
ROUTER_TEMPERATURE="${ROUTER_TEMPERATURE:-0.5}"
VISUAL_TOPK_RATIO="${VISUAL_TOPK_RATIO:-0.4}"
REASON_TOPK_RATIO="${REASON_TOPK_RATIO:-0.4}"
VISUAL_SPAN_START_RATIO="${VISUAL_SPAN_START_RATIO:-0.0}"
VISUAL_SPAN_END_RATIO="${VISUAL_SPAN_END_RATIO:-1.0}"
REASON_SPAN_START_RATIO="${REASON_SPAN_START_RATIO:-0.5}"
REASON_SPAN_END_RATIO="${REASON_SPAN_END_RATIO:-1.0}"
VISUAL_POLICY_WEIGHT="${VISUAL_POLICY_WEIGHT:-1.0}"
REASON_POLICY_WEIGHT="${REASON_POLICY_WEIGHT:-1.0}"
VISUAL_ANCHOR_EXTRA="${VISUAL_ANCHOR_EXTRA:-0.0}"
VISUAL_BRANCH_REWARD_MODE="${VISUAL_BRANCH_REWARD_MODE:-aver}"
VISUAL_ENGAGEMENT_TAU="${VISUAL_ENGAGEMENT_TAU:-0.2}"
VISUAL_ENGAGEMENT_LAYER_INDEX="${VISUAL_ENGAGEMENT_LAYER_INDEX:--1}"
VISUAL_ENGAGEMENT_MAX_SCORE_TOKENS="${VISUAL_ENGAGEMENT_MAX_SCORE_TOKENS:-128}"
VISUAL_ENGAGEMENT_SCORER_ATTN_IMPLEMENTATION="${VISUAL_ENGAGEMENT_SCORER_ATTN_IMPLEMENTATION:-eager}"
VISUAL_ENGAGEMENT_SCORER_DEVICE="${VISUAL_ENGAGEMENT_SCORER_DEVICE:-}"
MEASURE_RUNTIME="${MEASURE_RUNTIME:-0}"

args=(
  python src/main.py
  --dataset "${PATH_TO_DATA}"
  --dataset_split "${DATASET_SPLIT}"
  --model_name_or_path "${PATH_TO_MODEL}"
  --output_dir "${OUTPUT_DIR}"
  --attn_implementation "${ATTN_IMPLEMENTATION}"
  --start_data_idx "${START_DATA_IDX}"
  --end_data_idx "${END_DATA_IDX}"
  --k "${K}"
  --lr "${LR}"
  --max_num_steps "${MAX_NUM_STEPS}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --save_every "${SAVE_EVERY}"
  --solver_prompt_idx "${SOLVER_PROMPT_IDX}"
  --latent_start_strategy "${LATENT_START_STRATEGY}"
  --answer_tail_tokens "${ANSWER_TAIL_TOKENS}"
  --answer_start_fallback_ratio "${ANSWER_START_FALLBACK_RATIO}"
  --latent_anchor_weight "${LATENT_ANCHOR_WEIGHT}"
  --optimized_acceptance "${OPTIMIZED_ACCEPTANCE}"
  --reward_threshold "${REWARD_THRESHOLD}"
  --device "${DEVICE}"
  --logit_lr "${LOGIT_LR}"
  --noise_std "${NOISE_STD}"
  --noise_candidates "${NOISE_CANDIDATES}"
  --confidence_reward_mix "${CONFIDENCE_REWARD_MIX}"
  --baseline_mode "${BASELINE_MODE}"
  --baseline_num_candidates "${BASELINE_NUM_CANDIDATES}"
  --baseline_temperature "${BASELINE_TEMPERATURE}"
  --baseline_top_p "${BASELINE_TOP_P}"
  --original_latentseek_rule_mode "${ORIGINAL_LATENTSEEK_RULE_MODE}"
  --original_latentseek_verifier_max_new_tokens "${ORIGINAL_LATENTSEEK_VERIFIER_MAX_NEW_TOKENS}"
  --visual_reward_mode "${VISUAL_REWARD_MODE}"
  --visual_prior_weight "${VISUAL_PRIOR_WEIGHT}"
  --contrastive_temperature "${CONTRASTIVE_TEMPERATURE}"
  --candidate_aggregation "${CANDIDATE_AGGREGATION}"
  --aggregation_temperature "${AGGREGATION_TEMPERATURE}"
  --answer_finalizer_mode "${ANSWER_FINALIZER_MODE}"
  --answer_finalizer_max_new_tokens "${ANSWER_FINALIZER_MAX_NEW_TOKENS}"
  --visual_branch_reward_mode "${VISUAL_BRANCH_REWARD_MODE}"
  --visual_engagement_tau "${VISUAL_ENGAGEMENT_TAU}"
  --visual_engagement_layer_index "${VISUAL_ENGAGEMENT_LAYER_INDEX}"
  --visual_engagement_max_score_tokens "${VISUAL_ENGAGEMENT_MAX_SCORE_TOKENS}"
  --visual_engagement_scorer_attn_implementation "${VISUAL_ENGAGEMENT_SCORER_ATTN_IMPLEMENTATION}"
)

if [[ "${PROCESSOR_USE_FAST}" == "1" ]]; then args+=(--processor_use_fast); fi
if [[ -n "${QWEN_MIN_PIXELS}" ]]; then args+=(--qwen_min_pixels "${QWEN_MIN_PIXELS}"); fi
if [[ -n "${QWEN_MAX_PIXELS}" ]]; then args+=(--qwen_max_pixels "${QWEN_MAX_PIXELS}"); fi
if [[ -n "${MODEL_DEVICE_MAP}" ]]; then args+=(--model_device_map "${MODEL_DEVICE_MAP}"); fi
if [[ -n "${VISUAL_ENGAGEMENT_SCORER_DEVICE}" ]]; then args+=(--visual_engagement_scorer_device "${VISUAL_ENGAGEMENT_SCORER_DEVICE}"); fi
if [[ "${ALLOW_CUDNN_SDPA}" == "1" ]]; then args+=(--allow_cudnn_sdp); fi
if [[ "${MEASURE_RUNTIME}" == "1" ]]; then args+=(--measure_runtime); fi
if [[ "${RESUME}" == "1" ]]; then args+=(--resume); fi
if [[ "${USE_V1:-0}" == "1" ]]; then args+=(--use_v1); fi
if [[ "${NO_DUAL_SPACE:-0}" == "1" ]]; then args+=(--no_dual_space); fi
if [[ "${NO_CONFIDENCE_REWARD:-0}" == "1" ]]; then args+=(--no_confidence_reward); fi
if [[ "${ORIGINAL_LATENTSEEK_USE_COMPACT_VERIFIERS}" == "1" ]]; then args+=(--original_latentseek_use_compact_verifiers); fi
if [[ -n "${RULE_FORMAT_STRING:-}" ]]; then args+=(--rule_format_string "${RULE_FORMAT_STRING}"); fi

if [[ "${USE_EVIDENCE_CALIBRATION}" == "1" ]]; then
  args+=(--use_evidence_calibration --answer_reward_weight "${ANSWER_REWARD_WEIGHT}" --margin_reward_weight "${MARGIN_REWARD_WEIGHT}")
  if [[ "${USE_VISUAL_CLUES}" == "1" ]]; then
    args+=(--use_visual_clues --num_visual_clues "${NUM_VISUAL_CLUES}" --clue_reward_weight "${CLUE_REWARD_WEIGHT}" --clue_max_new_tokens "${CLUE_MAX_NEW_TOKENS}")
  fi
  if [[ "${USE_CROP_EVIDENCE}" == "1" ]]; then
    args+=(--use_crop_evidence --crop_reward_weight "${CROP_REWARD_WEIGHT}" --top_k_positive_crops "${TOP_K_POSITIVE_CROPS}" --max_negative_views "${MAX_NEGATIVE_VIEWS}" --crop_grid_sizes "${CROP_GRID_SIZES}" --max_crop_candidates_per_step "${MAX_CROP_CANDIDATES_PER_STEP}")
  fi
fi

if [[ "${USE_TOKEN_DISENTANGLE}" == "1" ]]; then
  args+=(
    --use_token_disentangle
    --token_router "${TOKEN_ROUTER}"
    --reason_router_signal "${REASON_ROUTER_SIGNAL}"
    --image_sensitivity_drop_prob "${IMAGE_SENSITIVITY_DROP_PROB}"
    --image_sensitivity_patch_size "${IMAGE_SENSITIVITY_PATCH_SIZE}"
    --visual_router_threshold "${VISUAL_ROUTER_THRESHOLD}"
    --entropy_router_threshold "${ENTROPY_ROUTER_THRESHOLD}"
    --router_temperature "${ROUTER_TEMPERATURE}"
    --visual_topk_ratio "${VISUAL_TOPK_RATIO}"
    --reason_topk_ratio "${REASON_TOPK_RATIO}"
    --visual_span_start_ratio "${VISUAL_SPAN_START_RATIO}"
    --visual_span_end_ratio "${VISUAL_SPAN_END_RATIO}"
    --reason_span_start_ratio "${REASON_SPAN_START_RATIO}"
    --reason_span_end_ratio "${REASON_SPAN_END_RATIO}"
    --visual_policy_weight "${VISUAL_POLICY_WEIGHT}"
    --reason_policy_weight "${REASON_POLICY_WEIGHT}"
    --visual_anchor_extra "${VISUAL_ANCHOR_EXTRA}"
  )
fi

if [[ "${DEBUG_VISUAL_FOCUS}" == "1" ]]; then args+=(--debug_visual_focus); fi
if [[ "${DEBUG_LOSS_TERMS}" == "1" ]]; then args+=(--debug_loss_terms); fi
if [[ "${DEBUG_REWARD_TERMS}" == "1" ]]; then args+=(--debug_reward_terms); fi
if [[ "${DEBUG_CANDIDATE_REWARDS}" == "1" ]]; then args+=(--debug_candidate_rewards); fi
if [[ "${DEBUG_CANDIDATE_CONDITIONED_CLUES}" == "1" ]]; then
  args+=(--debug_candidate_conditioned_clues --candidate_conditioned_clue_top_k "${CANDIDATE_CONDITIONED_CLUE_TOP_K}" --candidate_conditioned_clue_max_new_tokens "${CANDIDATE_CONDITIONED_CLUE_MAX_NEW_TOKENS}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "cd ${PROJECT_ROOT}"
  lt_print_command "${args[@]}"
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
"${args[@]}"
