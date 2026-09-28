import argparse
import json
import os
import random
import time

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

from evaluation.finalizer import finalize_answer, should_finalize_answer
from evaluation.visual_clues import generate_visual_clues
from rewards.confidence import ConfidenceReward
from rewards.evidence_calibration import EvidenceCalibratedReward
from data import get_dataset, prepare_example
from evaluation.extract_judge_answer import extract_answer, extract_true_answer, judge_answer, normalize_prediction
from benchmarks.hallusionbench import choose_hallusionbench_pairwise_prediction_likelihood
from benchmarks.logicvista import choose_logicvista_pairwise_prediction_likelihood
from benchmarks.mmstar import choose_mmstar_pairwise_prediction_by_letter_likelihood, resolve_mmstar_start_index
from benchmarks.mmmu import choose_mmmu_pairwise_prediction_by_letter_likelihood
from benchmarks.mathvista import choose_mathvista_pairwise_prediction_by_letter_likelihood, resolve_mathvista_start_index
from benchmarks.scienceqa import choose_scienceqa_pairwise_prediction_by_letter_likelihood
from methods.baselines import (
    OriginalLatentSeekRewardModel,
    best_of_n_visual_generation,
    legacy_latentseek_generation,
    original_latentseek_generation,
    reward_only_rerank_generation,
    self_consistency_generation,
)
from methods.latentgrad.optimize import optimized_generation, optimized_generation_v2
from methods.latentgrad.generate import original_generation
from benchmarks.realworldqa import choose_realworldqa_pairwise_prediction_likelihood
from models.internvl import InternVLProcessor, is_internvl_config, patch_internvl_generation
from rewards.visual_likelihood import VisualLikelihoodReward


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--dataset", type=str, default="openai/gsm8k", help="Dataset to evaluate")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--model_name_or_path", type=str, help="Path to the model")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--processor_use_fast", action="store_true", help="Use the fast AutoProcessor backend for Qwen2.5-VL")
    parser.add_argument("--qwen_min_pixels", type=int, default=None, help="Optional Qwen2.5-VL processor min_pixels override")
    parser.add_argument("--qwen_max_pixels", type=int, default=None, help="Optional Qwen2.5-VL processor max_pixels override")
    parser.add_argument(
        "--model_device_map",
        type=str,
        default=None,
        help='Optional transformers device_map override, e.g. "auto" for multi-GPU 32B inference.',
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        help="Transformer attention backend override",
    )
    parser.add_argument(
        "--allow_cudnn_sdp",
        action="store_true",
        help="Keep cuDNN SDPA enabled. By default it is disabled for Qwen2.5-VL stability on long generations.",
    )
    parser.add_argument("--start_data_idx", type=int, default=0, help="Start index of the data to evaluate")
    parser.add_argument("--end_data_idx", type=int, default=-1, help="End index of the data to evaluate, -1 means the full split")
    parser.add_argument("--solver_prompt_idx", type=int, default=0, help="Index of the solver prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping threshold")
    parser.add_argument("--k", type=float, default=0.1, help="Ratio of update length to the total length of hidden states")
    parser.add_argument("--max_num_steps", type=int, default=10, help="Number of optimization iterations")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Number of generated tokens")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--latent_start_strategy",
        type=str,
        default="auto",
        choices=["auto", "answer", "tail", "full"],
        help="Where to start latent optimization for the generated answer",
    )
    parser.add_argument("--answer_tail_tokens", type=int, default=0, help="How many tokens before the detected answer slot to also optimize")
    parser.add_argument("--answer_start_fallback_ratio", type=float, default=0.8, help="Fallback late-answer ratio when answer slot detection fails")
    parser.add_argument("--latent_anchor_weight", type=float, default=0.02, help="MSE trust-region weight toward the original multimodal latent states")
    parser.add_argument("--candidate_count", type=int, default=1, help="Number of answer-tail candidates to evaluate per latent update step")
    parser.add_argument("--candidate_topk", type=int, default=4, help="Top-k sampling pool for answer-tail candidate search")
    parser.add_argument("--candidate_temperature", type=float, default=1.0, help="Temperature for answer-tail candidate search")
    parser.add_argument(
        "--optimized_acceptance",
        type=str,
        default="always",
        choices=["always", "valid_prediction", "valid_prediction_mc_letter", "valid_prediction_pairwise"],
        help="Acceptance rule applied to optimized answers before final scoring/logging.",
    )
    parser.add_argument(
        "--use_v1",
        action="store_true",
        help="Use the legacy v1 optimizer instead of the default v2 pipeline",
    )
    parser.add_argument("--logit_lr", type=float, default=0.01, help="Learning rate for logit-space entropy minimization (v2)")
    parser.add_argument("--noise_std", type=float, default=0.05, help="Gaussian noise std for parallel trajectory search (v2)")
    parser.add_argument("--noise_candidates", type=int, default=3, help="Number of noisy candidates per optimization step (v2)")
    parser.add_argument("--dual_space", action="store_true", default=True, help="Enable dual-space optimization (v2)")
    parser.add_argument("--no_dual_space", action="store_true", help="Disable dual-space optimization (v2)")
    parser.add_argument("--use_confidence_reward", action="store_true", default=True, help="Enable the auxiliary confidence prior in v2")
    parser.add_argument("--no_confidence_reward", action="store_true", help="Disable the auxiliary confidence prior and rely on visual likelihood only")
    parser.add_argument("--confidence_reward_mix", type=float, default=0.7, help="Reserved confidence-prior mix coefficient for v2")
    parser.add_argument(
        "--visual_reward_mode",
        type=str,
        default="likelihood",
        choices=["likelihood", "contrastive"],
        help="Answer scoring objective for v2: plain visual likelihood or contrastive visual information gain",
    )
    parser.add_argument(
        "--visual_prior_weight",
        type=float,
        default=1.0,
        help="Weight on the question-only prior term in contrastive visual reward",
    )
    parser.add_argument(
        "--contrastive_temperature",
        type=float,
        default=1.0,
        help="Temperature used to squash contrastive visual reward into [-1, 1]",
    )
    parser.add_argument(
        "--use_evidence_calibration",
        action="store_true",
        help="Use ECL-TTS reward core: answer reward plus relative answer margin, with optional candidate-conditioned evidence verification.",
    )
    parser.add_argument(
        "--answer_reward_weight",
        type=float,
        default=1.0,
        help="Weight on the normalized answer reward inside evidence calibration.",
    )
    parser.add_argument(
        "--margin_reward_weight",
        type=float,
        default=0.5,
        help="Weight on the relative margin term inside evidence calibration.",
    )
    parser.add_argument(
        "--use_visual_clues",
        action="store_true",
        help="Generate 1-3 visual clues per example and use them to propose candidate-specific support regions.",
    )
    parser.add_argument(
        "--num_visual_clues",
        type=int,
        default=3,
        help="Maximum number of visual clues to generate per example.",
    )
    parser.add_argument(
        "--clue_reward_weight",
        type=float,
        default=0.3,
        help="Weight on the relative clue-margin term inside evidence calibration.",
    )
    parser.add_argument(
        "--clue_max_new_tokens",
        type=int,
        default=96,
        help="Maximum number of new tokens used when generating visual clues.",
    )
    parser.add_argument(
        "--use_crop_evidence",
        action="store_true",
        help="Build candidate-conditioned support/negative views and add counterfactual evidence gain into evidence calibration.",
    )
    parser.add_argument(
        "--crop_reward_weight",
        type=float,
        default=0.4,
        help="Weight on the candidate-conditioned crop verifier reward.",
    )
    parser.add_argument(
        "--top_k_positive_crops",
        type=int,
        default=2,
        help="How many supportive crop views to keep per sample.",
    )
    parser.add_argument(
        "--max_negative_views",
        type=int,
        default=3,
        help="How many negative crop or masked views to keep per sample.",
    )
    parser.add_argument(
        "--crop_grid_sizes",
        type=str,
        default="2,3",
        help="Comma-separated grid sizes used to build candidate crops, e.g. 2,3.",
    )
    parser.add_argument(
        "--max_crop_candidates_per_step",
        type=int,
        default=2,
        help="Only the top few candidates per step receive crop verification to control cost. Use 0 for all.",
    )
    parser.add_argument(
        "--candidate_aggregation",
        type=str,
        default="majority",
        choices=["majority", "reward_weighted"],
        help="How to aggregate candidate answers at the end of v2",
    )
    parser.add_argument(
        "--aggregation_temperature",
        type=float,
        default=0.5,
        help="Temperature for reward-weighted candidate aggregation",
    )
    parser.add_argument("--entropy_weight", type=float, default=0.4, help="Weight of entropy component in confidence reward (v2)")
    parser.add_argument("--logprob_weight", type=float, default=0.3, help="Weight of logprob component in confidence reward (v2)")
    parser.add_argument("--format_weight", type=float, default=0.2, help="Weight of format compliance in confidence reward (v2)")
    parser.add_argument("--visual_weight", type=float, default=0.1, help="Weight of visual consistency in confidence reward (v2)")
    parser.add_argument("--rule_format_string", type=str, default=None, help="the answer format that should follow")
    parser.add_argument("--reward_threshold", type=float, default=-0.2, help="Threshold for reward to stop optimization")
    parser.add_argument(
        "--debug_loss_terms",
        action="store_true",
        help="Print policy/logit/anchor/total loss terms at each v2 optimization step",
    )
    parser.add_argument(
        "--debug_reward_terms",
        action="store_true",
        help="Print detailed reward decomposition for the original answer and best candidate at each v2 step",
    )
    parser.add_argument(
        "--debug_candidate_rewards",
        action="store_true",
        help="Print every candidate's reward decomposition at each v2 step and save it into reward_diagnostics",
    )
    parser.add_argument(
        "--debug_visual_focus",
        action="store_true",
        help="Save visual-focus diagnostics for original and optimized latent tokens. Diagnostics only; does not change ECL-TTS reward.",
    )
    parser.add_argument(
        "--debug_candidate_conditioned_clues",
        action="store_true",
        help="Generate and save candidate-specific visual clue diagnostics for selected v2 candidates. Diagnostics only; does not change ECL-TTS reward.",
    )
    parser.add_argument(
        "--use_token_disentangle",
        action="store_true",
        help="Route answer/margin and visual-clue rewards to different latent-token subspaces.",
    )
    parser.add_argument(
        "--token_router",
        type=str,
        default="vppo_strict_topk",
        choices=[
            "visual_focus_entropy",
            "entropy_only",
            "visual_focus_only",
            "hard_topk",
            "image_sensitivity_topk",
            "image_sensitivity_overlap_topk",
            "image_sensitivity_swapped_topk",
            "vppo_sensitivity_topk",
            "perception_topk",
            "vppo_strict_topk",
            "vppo_overlap_topk",
            "vppo_swapped_topk",
            "raw_image_sensitivity_topk",
            "raw_image_sensitivity_overlap_topk",
            "raw_image_sensitivity_swapped_topk",
            "random_topk",
            "uniform_topk",
            "span_constrained",
            "hard_topk_span",
        ],
        help="Token routing rule used when --use_token_disentangle is enabled.",
    )
    parser.add_argument(
        "--reason_router_signal",
        type=str,
        default="entropy",
        choices=[
            "entropy",
            "neg_logprob",
            "token_loss",
            "random",
        ],
        help="Reasoning-token score used by hard/top-k token routers.",
    )
    parser.add_argument(
        "--image_sensitivity_drop_prob",
        type=float,
        default=0.5,
        help="Drop/blackening probability for image-sensitivity routing.",
    )
    parser.add_argument(
        "--image_sensitivity_patch_size",
        type=int,
        default=14,
        help="Raw-image patch size for VPPO strict image-sensitivity routing.",
    )
    parser.add_argument(
        "--visual_router_threshold",
        type=float,
        default=0.5,
        help="Threshold for visual-focus soft routing.",
    )
    parser.add_argument(
        "--entropy_router_threshold",
        type=float,
        default=0.0,
        help="Threshold for z-normalized entropy soft routing.",
    )
    parser.add_argument(
        "--router_temperature",
        type=float,
        default=0.5,
        help="Temperature for token routing sigmoid gates.",
    )
    parser.add_argument(
        "--visual_topk_ratio",
        type=float,
        default=0.4,
        help="Fraction of eligible update tokens selected as visual tokens for hard top-k token routing.",
    )
    parser.add_argument(
        "--reason_topk_ratio",
        type=float,
        default=0.4,
        help="Fraction of eligible update tokens selected as reasoning tokens for hard top-k token routing.",
    )
    parser.add_argument(
        "--visual_span_start_ratio",
        type=float,
        default=0.0,
        help="Relative start of the visual routing span inside the optimized latent slice.",
    )
    parser.add_argument(
        "--visual_span_end_ratio",
        type=float,
        default=1.0,
        help="Relative end of the visual routing span inside the optimized latent slice.",
    )
    parser.add_argument(
        "--reason_span_start_ratio",
        type=float,
        default=0.5,
        help="Relative start of the reasoning routing span inside the optimized latent slice.",
    )
    parser.add_argument(
        "--reason_span_end_ratio",
        type=float,
        default=1.0,
        help="Relative end of the reasoning routing span inside the optimized latent slice.",
    )
    parser.add_argument(
        "--visual_policy_weight",
        type=float,
        default=1.0,
        help="Multiplier for the visual routed policy-loss term.",
    )
    parser.add_argument(
        "--reason_policy_weight",
        type=float,
        default=1.0,
        help="Multiplier for the reasoning routed policy-loss term.",
    )
    parser.add_argument(
        "--visual_anchor_extra",
        type=float,
        default=0.0,
        help="Extra visual-token weighted MSE anchor when token disentangling is enabled.",
    )
    parser.add_argument(
        "--visual_branch_reward_mode",
        type=str,
        default="ecl",
        choices=["ecl", "aver", "visual_engagement", "attention_engagement", "answer_likelihood_delta", "image_likelihood_delta"],
        help="Visual-branch reward source for token disentangling: ECL clue/crop reward, AVER attention engagement, or answer likelihood delta.",
    )
    parser.add_argument(
        "--visual_engagement_tau",
        type=float,
        default=0.2,
        help="Temperature for AVER visual reward: tanh((E_current - E_original) / tau).",
    )
    parser.add_argument(
        "--visual_engagement_layer_index",
        type=str,
        default="-1",
        help="Attention layer used by AVER. -1 means final layer; use all/avg/mean to average all layers.",
    )
    parser.add_argument(
        "--visual_engagement_max_score_tokens",
        type=int,
        default=128,
        help="Maximum generated tokens scored by AVER per trace. Use 0 to score all generated tokens.",
    )
    parser.add_argument(
        "--visual_engagement_scorer_attn_implementation",
        type=str,
        default="eager",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        help="Attention backend for the AVER scorer model. Keep this eager unless the backend returns output_attentions.",
    )
    parser.add_argument(
        "--visual_engagement_scorer_device",
        type=str,
        default=None,
        help="Optional device for the separate AVER scorer model. Defaults to the main model device.",
    )
    parser.add_argument(
        "--candidate_conditioned_clue_top_k",
        type=int,
        default=2,
        help="How many candidates per optimization step receive candidate-conditioned clue diagnostics. Use 0 for all candidates.",
    )
    parser.add_argument(
        "--candidate_conditioned_clue_max_new_tokens",
        type=int,
        default=64,
        help="Maximum new tokens when generating candidate-conditioned visual clues.",
    )
    parser.add_argument("--save_every", type=int, default=1, help="Persist logistics and aggregated results every n samples")
    parser.add_argument(
        "--baseline_mode",
        type=str,
        default="none",
        choices=[
            "none",
            "self_consistency",
            "best_of_n_visual",
            "reward_only_rerank",
            "legacy_latentseek",
            "original_latentseek",
            "vanilla",
        ],
        help="Run a baseline method instead of the default latentgrad pipeline.",
    )
    parser.add_argument("--baseline_num_candidates", type=int, default=4, help="Number of sampled candidates for output-space baselines")
    parser.add_argument("--baseline_temperature", type=float, default=0.7, help="Sampling temperature for output-space baselines")
    parser.add_argument("--baseline_top_p", type=float, default=0.95, help="Top-p sampling for output-space baselines")
    parser.add_argument(
        "--original_latentseek_rule_mode",
        type=str,
        default="text",
        choices=["text", "multimodal"],
        help="Original LatentSeek ablation: upstream text rules only, or text rules plus visual grounding.",
    )
    parser.add_argument(
        "--original_latentseek_verifier_max_new_tokens",
        type=int,
        default=4096,
        help="Maximum verifier response length for original_latentseek.",
    )
    parser.add_argument(
        "--original_latentseek_use_compact_verifiers",
        action="store_true",
        help="Use compact one-line verifier prompts for original_latentseek text rules.",
    )
    parser.add_argument(
        "--answer_finalizer_mode",
        type=str,
        default="when_needed",
        choices=["off", "when_needed", "always"],
        help="Whether to run a second-stage answer finalizer",
    )
    parser.add_argument(
        "--answer_finalizer_max_new_tokens",
        type=int,
        default=64,
        help="Maximum finalizer response length",
    )
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    parser.add_argument(
        "--measure_runtime",
        action="store_true",
        help="Record synchronized per-example wall-clock time and CUDA peak memory.",
    )
    return parser.parse_args()


def _parse_int_csv(text, default=(2, 3)):
    if text is None:
        return tuple(int(item) for item in default)
    values = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = int(chunk)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    if values:
        return tuple(values)
    return tuple(int(item) for item in default)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def _ensure_jsonable(value):
    if isinstance(value, dict):
        return {key: _ensure_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_ensure_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_ensure_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _load_results(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _runtime_sync(enabled):
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def _runtime_now(enabled):
    _runtime_sync(enabled)
    return time.perf_counter()


def _runtime_reset_peak_memory(enabled):
    if enabled and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _runtime_peak_memory_bytes(enabled):
    if not enabled or not torch.cuda.is_available():
        return None
    _runtime_sync(enabled)
    return int(torch.cuda.max_memory_allocated())


QWEN_VL_MODEL_TYPES = {"qwen2_5_vl", "qwen3_vl"}
LLAVA_VL_MODEL_TYPES = {"llava", "llava_next", "llava_onevision", "llavaonevision1_5"}


def _configure_attention_runtime(config, allow_cudnn_sdp=False):
    if config.model_type not in QWEN_VL_MODEL_TYPES:
        return
    if allow_cudnn_sdp:
        return
    if not torch.cuda.is_available():
        return
    if not hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        return

    cudnn_enabled = True
    if hasattr(torch.backends.cuda, "cudnn_sdp_enabled"):
        cudnn_enabled = torch.backends.cuda.cudnn_sdp_enabled()
    if cudnn_enabled:
        torch.backends.cuda.enable_cudnn_sdp(False)
        print("[runtime] Disabled cuDNN SDPA for Qwen-VL stability on long multimodal generations.")


def _load_model_bundle(
    model_name_or_path,
    device,
    processor_use_fast=False,
    attn_implementation="auto",
    allow_cudnn_sdp=False,
    qwen_min_pixels=None,
    qwen_max_pixels=None,
    model_device_map=None,
):
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    _configure_attention_runtime(config, allow_cudnn_sdp=allow_cudnn_sdp)

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation
    if model_device_map:
        model_kwargs["device_map"] = model_device_map

    if config.model_type in QWEN_VL_MODEL_TYPES:
        model_cls = (
            Qwen3VLForConditionalGeneration
            if config.model_type == "qwen3_vl"
            else Qwen2_5_VLForConditionalGeneration
        )
        model = model_cls.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
        processor_kwargs = {"use_fast": processor_use_fast}
        if qwen_min_pixels is not None:
            processor_kwargs["min_pixels"] = qwen_min_pixels
        if qwen_max_pixels is not None:
            processor_kwargs["max_pixels"] = qwen_max_pixels
        processor = AutoProcessor.from_pretrained(model_name_or_path, **processor_kwargs)
        tokenizer = processor.tokenizer
    elif is_internvl_config(config):
        internvl_model_kwargs = dict(model_kwargs)
        internvl_model_kwargs.pop("attn_implementation", None)
        internvl_model_kwargs["low_cpu_mem_usage"] = False
        if attn_implementation in {"eager", "eager_paged"}:
            internvl_model_kwargs["use_flash_attn"] = False
        model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            **internvl_model_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            fix_mistral_regex=False,
        )
        processor = InternVLProcessor.from_pretrained(
            model_name_or_path,
            tokenizer=tokenizer,
            config=config,
        )
        model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        patch_internvl_generation(model)
    elif config.model_type in LLAVA_VL_MODEL_TYPES:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            **model_kwargs,
        )
        processor_kwargs = {"use_fast": processor_use_fast}
        if qwen_min_pixels is not None:
            processor_kwargs["min_pixels"] = qwen_min_pixels
        if qwen_max_pixels is not None:
            processor_kwargs["max_pixels"] = qwen_max_pixels
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            **processor_kwargs,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            **model_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        processor = None

    if not model_device_map:
        model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if processor is not None and processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    return model, tokenizer, processor


def _uses_visual_engagement_reward(mode):
    return str(mode or "ecl").strip().lower() in {
        "aver",
        "visual_engagement",
        "attention_engagement",
    }


def _load_visual_engagement_scorer(args, device, model, tokenizer, processor):
    if not _uses_visual_engagement_reward(args.visual_branch_reward_mode):
        return None, None, None, None

    scorer_device = args.visual_engagement_scorer_device or device
    scorer_attn = args.visual_engagement_scorer_attn_implementation
    if scorer_device == device and scorer_attn == args.attn_implementation:
        print(f"[AVER] Using main model for visual engagement scoring (attn={scorer_attn}, device={scorer_device}).")
        return model, tokenizer, processor, scorer_device

    print(
        "[AVER] Loading separate visual engagement scorer "
        f"(main_attn={args.attn_implementation}, scorer_attn={scorer_attn}, device={scorer_device})."
    )
    scorer_model, scorer_tokenizer, scorer_processor = _load_model_bundle(
        args.model_name_or_path,
        scorer_device,
        processor_use_fast=args.processor_use_fast,
        attn_implementation=scorer_attn,
        allow_cudnn_sdp=args.allow_cudnn_sdp,
        qwen_min_pixels=args.qwen_min_pixels,
        qwen_max_pixels=args.qwen_max_pixels,
        model_device_map=args.model_device_map,
    )
    for param in scorer_model.parameters():
        param.requires_grad_(False)
    return scorer_model, scorer_tokenizer, scorer_processor, scorer_device


def _build_summary(processed_total, evaluated_total, original_correct, optimized_correct, update_count, original_length, optimized_length, fitten_length):
    denom_processed = max(processed_total, 1)
    return {
        "processed_total": processed_total,
        "evaluated_total": evaluated_total,
        "original_accuracy": original_correct / evaluated_total if evaluated_total else None,
        "optimized_accuracy": optimized_correct / evaluated_total if evaluated_total else None,
        "average_update_length": update_count / denom_processed,
        "average_original_length": original_length / denom_processed,
        "average_optimized_length": optimized_length / denom_processed,
        "average_fitten_length": fitten_length / denom_processed,
        "original_correct": original_correct,
        "optimized_correct": optimized_correct,
    }


def _print_summary(summary):
    print(f"Processed total: {summary['processed_total']}")
    print(f"Evaluated total: {summary['evaluated_total']}")
    if summary["evaluated_total"]:
        print(f"Original accuracy: {summary['original_accuracy']:.4f}")
        print(f"Optimized accuracy: {summary['optimized_accuracy']:.4f}")
    else:
        print("Original accuracy: N/A (no public labels available)")
        print("Optimized accuracy: N/A (no public labels available)")
    print(f"Average update length: {summary['average_update_length']:.4f}")
    print(f"Average original length: {summary['average_original_length']:.4f}")
    print(f"Average optimized length: {summary['average_optimized_length']:.4f}")
    print(f"Average fitten length: {summary['average_fitten_length']:.4f}")


def _choose_pairwise_by_visual_reward(visual_reward, problem, media, original_prediction, optimized_prediction):
    original_score, original_visual_logprob, original_prior_logprob = visual_reward.score_short_answer(
        original_prediction,
        media=media,
        problem=problem,
    )
    optimized_score, optimized_visual_logprob, optimized_prior_logprob = visual_reward.score_short_answer(
        optimized_prediction,
        media=media,
        problem=problem,
    )
    selected = "optimized" if optimized_score > original_score else "original"
    return {
        "original_score": original_score,
        "optimized_score": optimized_score,
        "original_visual_logprob": original_visual_logprob,
        "optimized_visual_logprob": optimized_visual_logprob,
        "original_prior_logprob": original_prior_logprob,
        "optimized_prior_logprob": optimized_prior_logprob,
        "selected": selected,
        "reward_mode": getattr(visual_reward, "reward_mode", "likelihood"),
    }


def _resolve_start_index(args, tokenizer, original_output, hidden_states_list):
    original_length = len(hidden_states_list)
    if original_length <= 0:
        return 0

    if args.latent_start_strategy == "full":
        return 0
    if args.latent_start_strategy == "tail":
        return max(0, min(int(original_length * args.answer_start_fallback_ratio), original_length - 1))
    if "MathVista" in args.dataset and args.latent_start_strategy in {"auto", "answer"}:
        return resolve_mathvista_start_index(
            output_text=original_output,
            tokenizer=tokenizer,
            original_length=original_length,
            prompt_idx=args.solver_prompt_idx,
            answer_tail_tokens=args.answer_tail_tokens,
            fallback_ratio=args.answer_start_fallback_ratio,
        )
    if "MMStar" in args.dataset and args.latent_start_strategy in {"auto", "answer"}:
        return resolve_mmstar_start_index(
            output_text=original_output,
            tokenizer=tokenizer,
            original_length=original_length,
            prompt_idx=args.solver_prompt_idx,
            answer_tail_tokens=args.answer_tail_tokens,
            fallback_ratio=args.answer_start_fallback_ratio,
        )
    return 0


def main(args):
    if args.rule_format_string == "boxed":
        rule_format_string = r"\\boxed{(.*)}"
    else:
        if args.rule_format_string:
            raise ValueError("Unknown format")
        rule_format_string = None

    if args.seed:
        set_seed(args.seed)

    device = "cuda" if args.device is None and torch.cuda.is_available() else args.device or "cpu"

    model, tokenizer, processor = _load_model_bundle(
        args.model_name_or_path,
        device,
        processor_use_fast=args.processor_use_fast,
        attn_implementation=args.attn_implementation,
        allow_cudnn_sdp=args.allow_cudnn_sdp,
            qwen_min_pixels=args.qwen_min_pixels,
            qwen_max_pixels=args.qwen_max_pixels,
            model_device_map=args.model_device_map,
        )
    visual_engagement_model, visual_engagement_tokenizer, _visual_engagement_processor, visual_engagement_device = (
        _load_visual_engagement_scorer(args, device, model, tokenizer, processor)
    )

    confidence_reward = None
    visual_reward = None
    evidence_reward = None
    baseline_mode = getattr(args, "baseline_mode", "none")
    use_v2 = baseline_mode == "none" and not getattr(args, "use_v1", False)
    reward_model = None
    if baseline_mode == "legacy_latentseek":
        from rewards.reward import RewardModel

        reward_model = RewardModel(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            device=device,
            data_name=args.dataset,
            rule_format_string=rule_format_string,
        )
        print("[Baseline] legacy_latentseek with single-step verifier-reward latent update")
    elif baseline_mode == "original_latentseek":
        reward_model = OriginalLatentSeekRewardModel(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            device=device,
            data_name=args.dataset,
            rule_format_string=rule_format_string,
            rule_mode=args.original_latentseek_rule_mode,
            verifier_max_new_tokens=args.original_latentseek_verifier_max_new_tokens,
            use_compact_verifiers=args.original_latentseek_use_compact_verifiers,
        )
        print(
            "[Baseline] original_latentseek "
            f"rule_mode={args.original_latentseek_rule_mode}, "
            f"max_num_steps={args.max_num_steps}, k={args.k}, lr={args.lr}, "
            f"reward_threshold={args.reward_threshold}, "
            f"verifier_max_new_tokens={args.original_latentseek_verifier_max_new_tokens}, "
            f"compact_verifiers={args.original_latentseek_use_compact_verifiers}"
        )
    elif baseline_mode == "best_of_n_visual":
        if processor is not None:
            visual_reward = VisualLikelihoodReward(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                device=device,
                reward_mode=args.visual_reward_mode,
                prior_weight=args.visual_prior_weight,
                contrastive_temperature=args.contrastive_temperature,
            )
        print(
            f"[Baseline] best_of_n_visual num_candidates={args.baseline_num_candidates}, "
            f"temperature={args.baseline_temperature}, top_p={args.baseline_top_p}, "
            f"visual_reward={'yes' if visual_reward else 'no'}"
        )
    elif baseline_mode == "reward_only_rerank":
        if processor is not None:
            visual_reward = VisualLikelihoodReward(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                device=device,
                reward_mode=args.visual_reward_mode,
                prior_weight=args.visual_prior_weight,
                contrastive_temperature=args.contrastive_temperature,
            )
            if args.use_evidence_calibration:
                evidence_reward = EvidenceCalibratedReward(
                    visual_reward=visual_reward,
                    answer_weight=args.answer_reward_weight,
                    margin_weight=args.margin_reward_weight,
                    clue_weight=args.clue_reward_weight if args.use_visual_clues else 0.0,
                    crop_weight=args.crop_reward_weight if args.use_crop_evidence else 0.0,
                    max_crop_candidates_per_step=args.max_crop_candidates_per_step,
                )
        print(
            f"[Baseline] reward_only_rerank num_candidates={args.baseline_num_candidates}, "
            f"temperature={args.baseline_temperature}, top_p={args.baseline_top_p}, "
            f"visual_reward={'yes' if visual_reward else 'no'}, evidence_calibration={bool(evidence_reward)}, "
            f"answer_weight={args.answer_reward_weight}, margin_weight={args.margin_reward_weight}, "
            f"use_visual_clues={args.use_visual_clues}, clue_weight={args.clue_reward_weight}, "
            f"use_crop_evidence={args.use_crop_evidence}, crop_weight={args.crop_reward_weight}"
        )
    elif baseline_mode == "self_consistency":
        print(
            f"[Baseline] self_consistency num_candidates={args.baseline_num_candidates}, "
            f"temperature={args.baseline_temperature}, top_p={args.baseline_top_p}"
        )
    elif baseline_mode == "vanilla":
        print("[Baseline] vanilla direct generation only; no sampling, reranking, finalizer, or latent optimization.")
    elif use_v2:
        confidence_reward = ConfidenceReward(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            device=device,
            data_name=args.dataset,
            entropy_weight=args.entropy_weight,
            logprob_weight=args.logprob_weight,
            format_weight=args.format_weight,
            visual_weight=args.visual_weight,
        )
        if processor is not None:
            visual_reward = VisualLikelihoodReward(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                device=device,
                reward_mode=args.visual_reward_mode,
                prior_weight=args.visual_prior_weight,
                contrastive_temperature=args.contrastive_temperature,
            )
            if args.use_evidence_calibration:
                evidence_reward = EvidenceCalibratedReward(
                    visual_reward=visual_reward,
                    answer_weight=args.answer_reward_weight,
                    margin_weight=args.margin_reward_weight,
                    clue_weight=args.clue_reward_weight if args.use_visual_clues else 0.0,
                    crop_weight=args.crop_reward_weight if args.use_crop_evidence else 0.0,
                    max_crop_candidates_per_step=args.max_crop_candidates_per_step,
                )
        print(
            f"[LatentGrad v2] dual_space={args.dual_space and not args.no_dual_space}, "
            f"confidence_reward={args.use_confidence_reward and not args.no_confidence_reward}, "
            f"mix={args.confidence_reward_mix}, noise_std={args.noise_std}, "
            f"noise_candidates={args.noise_candidates}, visual_reward={'yes' if visual_reward else 'no'}, "
            f"visual_reward_mode={args.visual_reward_mode}, prior_weight={args.visual_prior_weight}, "
            f"evidence_calibration={bool(evidence_reward)}, "
            f"answer_weight={args.answer_reward_weight}, margin_weight={args.margin_reward_weight}, "
            f"use_visual_clues={args.use_visual_clues}, clue_weight={args.clue_reward_weight}, clue_mode=relative_margin, "
            f"num_visual_clues={args.num_visual_clues}, use_crop_evidence={args.use_crop_evidence}, "
            f"crop_weight={args.crop_reward_weight}, crop_mode=candidate_counterfactual, top_k_positive_crops={args.top_k_positive_crops}, "
            f"max_negative_views={args.max_negative_views}, max_crop_candidates_per_step={args.max_crop_candidates_per_step}, "
            f"token_disentangle={args.use_token_disentangle}, token_router={args.token_router}, "
            f"visual_router_threshold={args.visual_router_threshold}, entropy_router_threshold={args.entropy_router_threshold}, "
            f"router_temperature={args.router_temperature}, visual_topk_ratio={args.visual_topk_ratio}, "
            f"reason_topk_ratio={args.reason_topk_ratio}, visual_span=({args.visual_span_start_ratio},{args.visual_span_end_ratio}), "
            f"reason_span=({args.reason_span_start_ratio},{args.reason_span_end_ratio}), "
            f"visual_branch_reward_mode={args.visual_branch_reward_mode}, "
            f"visual_engagement_tau={args.visual_engagement_tau}, "
            f"visual_engagement_layer_index={args.visual_engagement_layer_index}, "
            f"visual_engagement_max_score_tokens={args.visual_engagement_max_score_tokens}"
        )
    else:
        print("[LatentGrad v1] legacy optimizer explicitly selected via --use_v1")
        from rewards.reward import RewardModel

        reward_model = RewardModel(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            device=device,
            data_name=args.dataset,
            rule_format_string=rule_format_string,
        )

    dataset = get_dataset(
        args.dataset,
        tokenizer=tokenizer,
        prompt_idx=args.solver_prompt_idx,
        dataset_split=args.dataset_split,
    )

    model_name = args.model_name_or_path.rstrip("/").split("/")[-1]
    data_name = args.dataset.rstrip("/").split("/")[-1]
    output_dir = f"{args.output_dir}/{model_name}-{data_name}-{args.dataset_split}-k{args.k}-lr{args.lr}-SolIdx{args.solver_prompt_idx}"
    if baseline_mode != "none":
        output_dir += (
            f"-BL{baseline_mode}"
            f"-N{args.baseline_num_candidates}"
            f"-T{args.baseline_temperature}"
            f"-P{args.baseline_top_p}"
        )
    elif use_v2 and args.use_evidence_calibration:
        output_dir += f"-ECLA{args.answer_reward_weight}-M{args.margin_reward_weight}"
        if args.visual_branch_reward_mode != "ecl":
            output_dir += (
                f"-VB{args.visual_branch_reward_mode}"
                f"-VT{args.visual_engagement_tau}"
                f"-VL{args.visual_engagement_layer_index}"
            )
        if args.use_visual_clues:
            output_dir += f"-CL{args.clue_reward_weight}-N{args.num_visual_clues}"
        if args.use_crop_evidence:
            output_dir += f"-CR{args.crop_reward_weight}-P{args.top_k_positive_crops}-NEG{args.max_negative_views}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)

    results_path = f"{output_dir}/results.json"
    logistics_path = f"{output_dir}/logistics.pt"
    summary_path = f"{output_dir}/summary.json"

    processed_total = 0
    evaluated_total = 0
    original_correct = 0
    optimized_correct = 0
    update_count = 0
    original_length = 0
    optimized_length = 0
    fitten_length = 0

    start_data_idx = max(0, args.start_data_idx)
    end_data_idx = len(dataset) if args.end_data_idx < 0 else min(args.end_data_idx, len(dataset))

    aggregated_results = _load_results(results_path)
    if args.resume and os.path.exists(logistics_path):
        print(f"Resume from {output_dir}")
        logistics = torch.load(logistics_path)
        start_data_idx = logistics["start_idx"]
        original_correct = logistics["original_correct"]
        optimized_correct = logistics["optimized_correct"]
        processed_total = logistics.get("processed_total", logistics.get("total", 0))
        evaluated_total = logistics.get("evaluated_total", logistics.get("total", 0))
        update_count = logistics["update_count"]
        original_length = logistics["original_length"]
        optimized_length = logistics["optimized_length"]
        fitten_length = logistics["fitten_length"]

    print(f"Start to evaluate {args.dataset} [{args.dataset_split}] from {start_data_idx} to {end_data_idx}...")

    for i in range(start_data_idx, end_data_idx):
        _runtime_reset_peak_memory(args.measure_runtime)
        example_started_at = _runtime_now(args.measure_runtime)
        example = dataset[i]
        prepared = prepare_example(
            example=example,
            data_name_or_path=args.dataset,
            tokenizer=tokenizer,
            prompt_idx=args.solver_prompt_idx,
            processor=processor,
        )
        true_answer = extract_true_answer(prepared["answer"], name=args.dataset) if prepared["answer"] is not None else None
        problem = prepared.get("problem")
        media = None
        if "images" in prepared:
            media = {"images": prepared["images"]}
            if prepared["images"]:
                media["image"] = prepared["images"][0]
        elif "image" in prepared:
            media = {"image": prepared["image"], "images": [prepared["image"]]}

        print(f"Question: {prepared['question']}")
        print(f"True answer: {true_answer}")

        initial_generation_started_at = _runtime_now(args.measure_runtime)
        original_output, hidden_states_list, input_ids = original_generation(
            input_text=prepared["formatted"],
            prompt_inputs=prepared["model_inputs"],
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        initial_generation_finished_at = _runtime_now(args.measure_runtime)
        start_index = _resolve_start_index(args, tokenizer, original_output, hidden_states_list)

        baseline_info = None
        reward_diagnostics = None
        visual_clues = []
        crop_evidence = None
        method_started_at = _runtime_now(args.measure_runtime)
        if baseline_mode == "vanilla":
            optimized_output = original_output
            reward_history = [0.0]
            baseline_info = {"mode": "vanilla"}
            new_original_length = len(hidden_states_list)
            new_optimized_length = len(tokenizer.encode(optimized_output))
            new_update_length = 0
        elif baseline_mode == "self_consistency":
            optimized_output, reward_history, baseline_info = self_consistency_generation(
                model=model,
                tokenizer=tokenizer,
                device=device,
                input_text=prepared["formatted"],
                prompt_inputs=prepared["model_inputs"],
                max_new_tokens=args.max_new_tokens,
                data_name=args.dataset,
                prompt_idx=args.solver_prompt_idx,
                model_name_or_path=args.model_name_or_path,
                problem=problem,
                num_candidates=args.baseline_num_candidates,
                temperature=args.baseline_temperature,
                top_p=args.baseline_top_p,
            )
            new_original_length = len(hidden_states_list)
            new_optimized_length = len(tokenizer.encode(optimized_output))
            new_update_length = 0
        elif baseline_mode == "best_of_n_visual":
            optimized_output, reward_history, baseline_info = best_of_n_visual_generation(
                model=model,
                tokenizer=tokenizer,
                device=device,
                input_text=prepared["formatted"],
                prompt_inputs=prepared["model_inputs"],
                question=prepared["question"],
                media=media,
                problem=problem,
                visual_reward=visual_reward,
                max_new_tokens=args.max_new_tokens,
                data_name=args.dataset,
                prompt_idx=args.solver_prompt_idx,
                model_name_or_path=args.model_name_or_path,
                num_candidates=args.baseline_num_candidates,
                temperature=args.baseline_temperature,
                top_p=args.baseline_top_p,
            )
            new_original_length = len(hidden_states_list)
            new_optimized_length = len(tokenizer.encode(optimized_output))
            new_update_length = 0
        elif baseline_mode == "reward_only_rerank":
            if evidence_reward is not None and args.use_visual_clues:
                visual_clues = generate_visual_clues(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    device=device,
                    problem=problem,
                    media=media,
                    draft_response=None,
                    max_new_tokens=args.clue_max_new_tokens,
                    num_clues=args.num_visual_clues,
                )
                if visual_clues:
                    print(f"-- Visual clues: {visual_clues}")
                else:
                    print("-- Visual clues: none")
            if evidence_reward is not None and args.use_crop_evidence:
                crop_evidence = evidence_reward.prepare_crop_evidence(
                    question=prepared["question"],
                    clues=visual_clues,
                    media=media,
                    problem=problem,
                    top_k_positive=args.top_k_positive_crops,
                    max_negative=args.max_negative_views,
                    grid_sizes=_parse_int_csv(args.crop_grid_sizes),
                )
                crop_summary = None if crop_evidence is None else crop_evidence.get("summary")
                if crop_summary is not None:
                    print(
                        "-- Crop evidence: "
                        f"positive={len(crop_summary.get('positive_views') or [])}, "
                        f"negative={len(crop_summary.get('negative_views') or [])}, "
                        f"candidate_crops={crop_summary.get('num_candidate_crops')}"
                    )
                else:
                    print("-- Crop evidence: none")
            optimized_output, reward_history, baseline_info = reward_only_rerank_generation(
                model=model,
                tokenizer=tokenizer,
                device=device,
                input_text=prepared["formatted"],
                prompt_inputs=prepared["model_inputs"],
                question=prepared["question"],
                media=media,
                problem=problem,
                visual_reward=visual_reward,
                evidence_reward=evidence_reward,
                visual_clues=visual_clues,
                crop_evidence=crop_evidence,
                original_answer=original_output,
                max_new_tokens=args.max_new_tokens,
                data_name=args.dataset,
                prompt_idx=args.solver_prompt_idx,
                model_name_or_path=args.model_name_or_path,
                num_candidates=args.baseline_num_candidates,
                temperature=args.baseline_temperature,
                top_p=args.baseline_top_p,
            )
            new_original_length = len(hidden_states_list)
            new_optimized_length = len(tokenizer.encode(optimized_output))
            new_update_length = 0
        elif baseline_mode == "legacy_latentseek":
            optimized_output, reward_history, new_original_length, new_optimized_length, new_update_length, baseline_info = legacy_latentseek_generation(
                reward_model=reward_model,
                model=model,
                tokenizer=tokenizer,
                device=device,
                question=prepared["question"],
                input_text=prepared["formatted"],
                original_answer=original_output,
                original_hidden_states_list=hidden_states_list,
                input_ids=input_ids,
                start_index=start_index,
                prompt_inputs=prepared["model_inputs"],
                media=media,
                problem=problem,
                max_new_tokens=args.max_new_tokens,
                lr=args.lr,
                grad_clip=args.grad_clip,
                k=args.k,
                latent_start_strategy=args.latent_start_strategy,
            )
        elif baseline_mode == "original_latentseek":
            optimized_output, reward_history, new_original_length, new_optimized_length, new_update_length, baseline_info = original_latentseek_generation(
                reward_model=reward_model,
                model=model,
                tokenizer=tokenizer,
                device=device,
                question=prepared["question"],
                input_text=prepared["formatted"],
                original_answer=original_output,
                original_hidden_states_list=hidden_states_list,
                input_ids=input_ids,
                start_index=start_index,
                prompt_inputs=prepared["model_inputs"],
                media=media,
                problem=problem,
                max_num_steps=args.max_num_steps,
                max_new_tokens=args.max_new_tokens,
                lr=args.lr,
                grad_clip=args.grad_clip,
                k=args.k,
                reward_threshold=args.reward_threshold,
                latent_start_strategy=args.latent_start_strategy,
            )
        elif use_v2:
            if evidence_reward is not None and args.use_visual_clues:
                visual_clues = generate_visual_clues(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    device=device,
                    problem=problem,
                    media=media,
                    draft_response=None,
                    max_new_tokens=args.clue_max_new_tokens,
                    num_clues=args.num_visual_clues,
                )
                if visual_clues:
                    print(f"-- Visual clues: {visual_clues}")
                else:
                    print("-- Visual clues: none")
            if evidence_reward is not None and args.use_crop_evidence:
                crop_evidence = evidence_reward.prepare_crop_evidence(
                    question=prepared["question"],
                    clues=visual_clues,
                    media=media,
                    problem=problem,
                    top_k_positive=args.top_k_positive_crops,
                    max_negative=args.max_negative_views,
                    grid_sizes=_parse_int_csv(args.crop_grid_sizes),
                )
                crop_summary = None if crop_evidence is None else crop_evidence.get("summary")
                if crop_summary is not None:
                    print(
                        "-- Crop evidence: "
                        f"positive={len(crop_summary.get('positive_views') or [])}, "
                        f"negative={len(crop_summary.get('negative_views') or [])}, "
                        f"candidate_crops={crop_summary.get('num_candidate_crops')}"
                    )
                else:
                    print("-- Crop evidence: none")
            reward_diagnostics = {}
            optimized_output, reward_history, new_original_length, new_optimized_length, new_update_length = optimized_generation_v2(
                confidence_reward=confidence_reward,
                reward_model=reward_model,
                model=model,
                tokenizer=tokenizer,
                device=device,
                question=prepared["question"],
                input_text=prepared["formatted"],
                original_answer=original_output,
                original_hidden_states_list=hidden_states_list,
                input_ids=input_ids,
                start_index=start_index,
                prompt_inputs=prepared["model_inputs"],
                processor=processor,
                media=media,
                problem=problem,
                max_num_steps=args.max_num_steps,
                max_new_tokens=args.max_new_tokens,
                lr=args.lr,
                grad_clip=args.grad_clip,
                k=args.k,
                reward_threshold=args.reward_threshold,
                anchor_weight=args.latent_anchor_weight,
                logit_lr=args.logit_lr,
                noise_std=args.noise_std,
                noise_candidates=args.noise_candidates,
                dual_space=args.dual_space and not args.no_dual_space,
                use_confidence_reward=args.use_confidence_reward and not args.no_confidence_reward,
                confidence_reward_mix=args.confidence_reward_mix,
                visual_reward=visual_reward,
                evidence_reward=evidence_reward,
                visual_clues=visual_clues,
                crop_evidence=crop_evidence,
                candidate_aggregation=args.candidate_aggregation,
                aggregation_temperature=args.aggregation_temperature,
                debug_loss_terms=args.debug_loss_terms,
                debug_reward_terms=args.debug_reward_terms,
                debug_candidate_rewards=args.debug_candidate_rewards,
                debug_visual_focus=args.debug_visual_focus,
                debug_candidate_conditioned_clues=args.debug_candidate_conditioned_clues,
                candidate_conditioned_clue_top_k=args.candidate_conditioned_clue_top_k,
                candidate_conditioned_clue_max_new_tokens=args.candidate_conditioned_clue_max_new_tokens,
                use_token_disentangle=args.use_token_disentangle,
                token_router=args.token_router,
                reason_router_signal=args.reason_router_signal,
                image_sensitivity_drop_prob=args.image_sensitivity_drop_prob,
                image_sensitivity_patch_size=args.image_sensitivity_patch_size,
                visual_router_threshold=args.visual_router_threshold,
                entropy_router_threshold=args.entropy_router_threshold,
                router_temperature=args.router_temperature,
                visual_topk_ratio=args.visual_topk_ratio,
                reason_topk_ratio=args.reason_topk_ratio,
                visual_span_start_ratio=args.visual_span_start_ratio,
                visual_span_end_ratio=args.visual_span_end_ratio,
                reason_span_start_ratio=args.reason_span_start_ratio,
                reason_span_end_ratio=args.reason_span_end_ratio,
                visual_policy_weight=args.visual_policy_weight,
                reason_policy_weight=args.reason_policy_weight,
                visual_anchor_extra=args.visual_anchor_extra,
                visual_branch_reward_mode=args.visual_branch_reward_mode,
                visual_engagement_tau=args.visual_engagement_tau,
                visual_engagement_layer_index=args.visual_engagement_layer_index,
                visual_engagement_max_score_tokens=args.visual_engagement_max_score_tokens,
                visual_engagement_model=visual_engagement_model,
                visual_engagement_tokenizer=visual_engagement_tokenizer,
                visual_engagement_device=visual_engagement_device,
                reward_diagnostics=reward_diagnostics,
            )
        else:
            optimized_output, reward_history, new_original_length, new_optimized_length, new_update_length = optimized_generation(
                reward_model=reward_model,
                model=model,
                tokenizer=tokenizer,
                device=device,
                question=prepared["question"],
                input_text=prepared["formatted"],
                original_answer=original_output,
                original_hidden_states_list=hidden_states_list,
                input_ids=input_ids,
                start_index=start_index,
                prompt_inputs=prepared["model_inputs"],
                media=media,
                problem=problem,
                max_num_steps=args.max_num_steps,
                max_new_tokens=args.max_new_tokens,
                lr=args.lr,
                grad_clip=args.grad_clip,
                k=args.k,
                reward_threshold=args.reward_threshold,
                anchor_weight=args.latent_anchor_weight,
                candidate_count=args.candidate_count,
                candidate_topk=args.candidate_topk,
                candidate_temperature=args.candidate_temperature,
            )

        method_finished_at = _runtime_now(args.measure_runtime)

        raw_original_output = original_output
        raw_optimized_output = optimized_output

        update_count += len(reward_history) - 1
        original_length += new_original_length
        optimized_length += new_optimized_length
        fitten_length += (new_optimized_length - new_update_length) if len(reward_history) > 1 else 0

        original_extraction = extract_answer(original_output, data_name=args.dataset, prompt_idx=args.solver_prompt_idx, model_name=args.model_name_or_path, problem=problem)
        optimized_extraction = extract_answer(optimized_output, data_name=args.dataset, prompt_idx=args.solver_prompt_idx, model_name=args.model_name_or_path, problem=problem)

        original_prediction = normalize_prediction(original_extraction, data_name=args.dataset, problem=problem) if original_extraction is not None else None
        optimized_prediction = normalize_prediction(optimized_extraction, data_name=args.dataset, problem=problem) if optimized_extraction is not None else None

        original_finalizer = None
        optimized_finalizer = None
        answer_finalizer_mode = "off" if baseline_mode == "vanilla" else args.answer_finalizer_mode
        if should_finalize_answer(
            response_text=original_output,
            extraction=original_extraction,
            normalized_prediction=original_prediction,
            data_name=args.dataset,
            prompt_idx=args.solver_prompt_idx,
            mode=answer_finalizer_mode,
        ):
            original_output, original_finalizer = finalize_answer(
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                device=device,
                data_name=args.dataset,
                prompt_idx=args.solver_prompt_idx,
                prepared=prepared,
                draft_response=original_output,
                max_new_tokens=args.answer_finalizer_max_new_tokens,
            )
            original_extraction = extract_answer(
                original_output,
                data_name=args.dataset,
                prompt_idx=args.solver_prompt_idx,
                model_name=args.model_name_or_path,
                problem=problem,
            )
            original_prediction = normalize_prediction(original_extraction, data_name=args.dataset, problem=problem) if original_extraction is not None else None

        if should_finalize_answer(
            response_text=optimized_output,
            extraction=optimized_extraction,
            normalized_prediction=optimized_prediction,
            data_name=args.dataset,
            prompt_idx=args.solver_prompt_idx,
            mode=answer_finalizer_mode,
        ):
            optimized_output, optimized_finalizer = finalize_answer(
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                device=device,
                data_name=args.dataset,
                prompt_idx=args.solver_prompt_idx,
                prepared=prepared,
                draft_response=optimized_output,
                max_new_tokens=args.answer_finalizer_max_new_tokens,
            )
            optimized_extraction = extract_answer(
                optimized_output,
                data_name=args.dataset,
                prompt_idx=args.solver_prompt_idx,
                model_name=args.model_name_or_path,
                problem=problem,
            )
            optimized_prediction = normalize_prediction(optimized_extraction, data_name=args.dataset, problem=problem) if optimized_extraction is not None else None

        original_correct_add = None
        optimized_correct_add = None
        if true_answer is not None:
            original_correct_add = judge_answer(original_output, true_answer, data_name=args.dataset, prompt_idx=args.solver_prompt_idx, model_name=args.model_name_or_path, problem=problem)
            optimized_correct_add = judge_answer(optimized_output, true_answer, data_name=args.dataset, prompt_idx=args.solver_prompt_idx, model_name=args.model_name_or_path, problem=problem)

        accepted_optimized = True
        acceptance_info = {"mode": args.optimized_acceptance}
        if baseline_mode != "none":
            acceptance_info = {"mode": "baseline_bypass"}
        elif args.optimized_acceptance in {"valid_prediction", "valid_prediction_mc_letter"} and optimized_prediction is None:
            accepted_optimized = False
            optimized_output = original_output
            optimized_extraction = original_extraction
            optimized_prediction = original_prediction
            optimized_correct_add = original_correct_add
            acceptance_info["reason"] = "optimized_prediction_none"
        elif (
            args.optimized_acceptance == "valid_prediction_mc_letter"
            and ("MathVista" in args.dataset or "MMStar" in args.dataset or "MMMU" in args.dataset or "ScienceQA" in args.dataset)
            and processor is not None
            and problem is not None
            and problem.get("question_type") == "multi_choice"
            and original_prediction is not None
            and optimized_prediction is not None
            and original_prediction != optimized_prediction
            and media is not None
            and (media.get("image") is not None or media.get("images"))
        ):
            if visual_reward is not None and getattr(visual_reward, "reward_mode", "likelihood") == "contrastive":
                pairwise_decision = _choose_pairwise_by_visual_reward(
                    visual_reward=visual_reward,
                    problem=problem,
                    media=media,
                    original_prediction=original_prediction,
                    optimized_prediction=optimized_prediction,
                )
            else:
                if "MMStar" in args.dataset:
                    pairwise_decision = choose_mmstar_pairwise_prediction_by_letter_likelihood(
                        model=model,
                        processor=processor,
                        image=media["image"],
                        problem=problem,
                        original_prediction=original_prediction,
                        optimized_prediction=optimized_prediction,
                    )
                elif "MMMU" in args.dataset:
                    pairwise_decision = choose_mmmu_pairwise_prediction_by_letter_likelihood(
                        model=model,
                        processor=processor,
                        image=media["image"],
                        problem=problem,
                        original_prediction=original_prediction,
                        optimized_prediction=optimized_prediction,
                    )
                elif "ScienceQA" in args.dataset:
                    pairwise_decision = choose_scienceqa_pairwise_prediction_by_letter_likelihood(
                        model=model,
                        processor=processor,
                        image=media["image"],
                        problem=problem,
                        original_prediction=original_prediction,
                        optimized_prediction=optimized_prediction,
                    )
                else:
                    pairwise_decision = choose_mathvista_pairwise_prediction_by_letter_likelihood(
                        model=model,
                        processor=processor,
                        image=media["image"],
                        problem=problem,
                        original_prediction=original_prediction,
                        optimized_prediction=optimized_prediction,
                    )
            acceptance_info.update(pairwise_decision)
            if pairwise_decision.get("selected") == "original":
                accepted_optimized = False
                optimized_output = original_output
                optimized_extraction = original_extraction
                optimized_prediction = original_prediction
                optimized_correct_add = original_correct_add
        elif (
            args.optimized_acceptance == "valid_prediction_pairwise"
            and "LogicVista" in args.dataset
            and processor is not None
            and problem is not None
            and original_prediction is not None
            and optimized_prediction is not None
            and original_prediction != optimized_prediction
            and media is not None
            and media.get("image") is not None
        ):
            if visual_reward is not None and getattr(visual_reward, "reward_mode", "likelihood") == "contrastive":
                pairwise_decision = _choose_pairwise_by_visual_reward(
                    visual_reward=visual_reward,
                    problem=problem,
                    media=media,
                    original_prediction=original_prediction,
                    optimized_prediction=optimized_prediction,
                )
            else:
                pairwise_decision = choose_logicvista_pairwise_prediction_likelihood(
                    model=model,
                    processor=processor,
                    image=media["image"],
                    problem=problem,
                    original_prediction=original_prediction,
                    optimized_prediction=optimized_prediction,
                )
            acceptance_info.update(pairwise_decision)
            if pairwise_decision.get("selected") == "original":
                accepted_optimized = False
                optimized_output = original_output
                optimized_extraction = original_extraction
                optimized_prediction = original_prediction
                optimized_correct_add = original_correct_add
        elif (
            args.optimized_acceptance == "valid_prediction_pairwise"
            and "RealWorldQA" in args.dataset
            and processor is not None
            and problem is not None
            and original_prediction is not None
            and optimized_prediction is not None
            and original_prediction != optimized_prediction
            and media is not None
            and media.get("image") is not None
        ):
            if visual_reward is not None and getattr(visual_reward, "reward_mode", "likelihood") == "contrastive":
                pairwise_decision = _choose_pairwise_by_visual_reward(
                    visual_reward=visual_reward,
                    problem=problem,
                    media=media,
                    original_prediction=original_prediction,
                    optimized_prediction=optimized_prediction,
                )
            else:
                pairwise_decision = choose_realworldqa_pairwise_prediction_likelihood(
                    model=model,
                    processor=processor,
                    image=media["image"],
                    problem=problem,
                    original_prediction=original_prediction,
                    optimized_prediction=optimized_prediction,
                )
            acceptance_info.update(pairwise_decision)
            if pairwise_decision.get("selected") == "original":
                accepted_optimized = False
                optimized_output = original_output
                optimized_extraction = original_extraction
                optimized_prediction = original_prediction
                optimized_correct_add = original_correct_add
        elif (
            args.optimized_acceptance == "valid_prediction_pairwise"
            and "HallusionBench" in args.dataset
            and processor is not None
            and problem is not None
            and original_prediction is not None
            and optimized_prediction is not None
            and original_prediction != optimized_prediction
        ):
            if visual_reward is not None and getattr(visual_reward, "reward_mode", "likelihood") == "contrastive":
                pairwise_decision = _choose_pairwise_by_visual_reward(
                    visual_reward=visual_reward,
                    problem=problem,
                    media=media or {},
                    original_prediction=original_prediction,
                    optimized_prediction=optimized_prediction,
                )
            else:
                pairwise_decision = choose_hallusionbench_pairwise_prediction_likelihood(
                    model=model,
                    processor=processor,
                    media=media or {},
                    problem=problem,
                    original_prediction=original_prediction,
                    optimized_prediction=optimized_prediction,
                )
            acceptance_info.update(pairwise_decision)
            if pairwise_decision.get("selected") == "original":
                accepted_optimized = False
                optimized_output = original_output
                optimized_extraction = original_extraction
                optimized_prediction = original_prediction
                optimized_correct_add = original_correct_add
        if true_answer is not None:
            original_correct += int(original_correct_add)
            optimized_correct += int(optimized_correct_add)
            evaluated_total += 1

        processed_total += 1
        example_finished_at = _runtime_now(args.measure_runtime)
        runtime_info = None
        if args.measure_runtime:
            runtime_info = {
                "total_seconds": float(example_finished_at - example_started_at),
                "initial_generation_seconds": float(initial_generation_finished_at - initial_generation_started_at),
                "method_seconds": float(method_finished_at - method_started_at),
                "postprocess_seconds": float(example_finished_at - method_finished_at),
                "peak_cuda_memory_bytes": _runtime_peak_memory_bytes(True),
            }

        result_key = prepared["example_id"] if prepared["example_id"] is not None else str(i)
        result_record = {
            "index": i,
            "example_id": result_key,
            "dataset": args.dataset,
            "dataset_split": args.dataset_split,
            "question": prepared["question"],
            "answer": true_answer,
            "original_raw_response": raw_original_output,
            "optimized_raw_response": raw_optimized_output,
            "original_response": original_output,
            "optimized_response": optimized_output,
            "response": optimized_output,
            "original_extraction": original_extraction,
            "optimized_extraction": optimized_extraction,
            "extraction": optimized_extraction,
            "original_prediction": original_prediction,
            "optimized_prediction": optimized_prediction,
            "prediction": optimized_prediction,
            "original_true_false": original_correct_add,
            "optimized_true_false": optimized_correct_add,
            "true_false": optimized_correct_add,
            "reward_history": [float(item) for item in reward_history],
            "update_steps": len(reward_history) - 1,
            "accepted_optimized": accepted_optimized,
            "acceptance_info": acceptance_info,
            "use_evidence_calibration": bool(evidence_reward),
            "requested_visual_clues": bool(args.use_visual_clues),
            "use_visual_clues": bool(visual_clues),
            "visual_clues": list(visual_clues),
            "requested_crop_evidence": bool(args.use_crop_evidence),
            "use_crop_evidence": crop_evidence is not None,
            "crop_evidence": None if crop_evidence is None else _ensure_jsonable(crop_evidence.get("summary")),
            "start_index": start_index,
            "original_length": new_original_length,
            "optimized_length": new_optimized_length,
            "update_length": new_update_length,
            "answer_finalizer_mode": answer_finalizer_mode,
            "original_finalizer_used": original_finalizer is not None,
            "optimized_finalizer_used": optimized_finalizer is not None,
        }
        if runtime_info is not None:
            result_record["runtime"] = runtime_info
        if original_finalizer is not None:
            result_record["original_finalizer"] = original_finalizer
        if optimized_finalizer is not None:
            result_record["optimized_finalizer"] = optimized_finalizer
        if problem is not None:
            result_record.update(_ensure_jsonable(problem))
        if baseline_info is not None:
            result_record["baseline_info"] = _ensure_jsonable(baseline_info)
        if reward_diagnostics:
            result_record["reward_diagnostics"] = _ensure_jsonable(reward_diagnostics)

        aggregated_results[result_key] = result_record
        _save_json(f"{output_dir}/test/{result_key}.json", result_record)

        torch.save(
            {
                "original_correct": original_correct,
                "optimized_correct": optimized_correct,
                "processed_total": processed_total,
                "evaluated_total": evaluated_total,
                "start_idx": i + 1,
                "update_count": update_count,
                "original_length": original_length,
                "optimized_length": optimized_length,
                "fitten_length": fitten_length,
            },
            logistics_path,
        )

        if (processed_total % args.save_every == 0) or (i == end_data_idx - 1):
            _save_json(results_path, aggregated_results)
            summary = _build_summary(processed_total, evaluated_total, original_correct, optimized_correct, update_count, original_length, optimized_length, fitten_length)
            _save_json(summary_path, summary)

    summary = _build_summary(processed_total, evaluated_total, original_correct, optimized_correct, update_count, original_length, optimized_length, fitten_length)
    _print_summary(summary)
    return summary


if __name__ == "__main__":
    args = parse_args()
    for arg in vars(args):
        print(f"-- {arg}: {getattr(args, arg)}")
    main(args)
