import math

import numpy as np
import torch
from PIL import Image
from evaluation.visual_clues import generate_candidate_visual_clues
from rewards.visual_engagement import (
    strip_stop_tokens,
    visual_engagement_reward,
    visual_engagement_score,
)

from methods.latentgrad.generation_utils import (
    GENERATION_MEDIA_KEYS,
    build_generation_kwargs,
    get_stop_token_ids,
    move_model_inputs,
    reset_multimodal_generation_state,
    trim_after_stop_words,
)

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

IMAGE_SENSITIVITY_ROUTERS = {
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
}
STRICT_VPPO_IMAGE_SENSITIVITY_ROUTERS = {
    "vppo_strict_topk",
    "vppo_overlap_topk",
    "vppo_swapped_topk",
    "raw_image_sensitivity_topk",
    "raw_image_sensitivity_overlap_topk",
    "raw_image_sensitivity_swapped_topk",
}
OVERLAP_IMAGE_SENSITIVITY_ROUTERS = {
    "image_sensitivity_overlap_topk",
    "vppo_overlap_topk",
    "raw_image_sensitivity_overlap_topk",
}
SWAPPED_IMAGE_SENSITIVITY_ROUTERS = {
    "image_sensitivity_swapped_topk",
    "vppo_swapped_topk",
    "raw_image_sensitivity_swapped_topk",
}


def _preview_text(text, limit=80):
    text = "" if text is None else str(text).strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _format_evidence_reward_breakdown(reward_info, evidence_reward=None):
    if reward_info is None:
        return "reward_info=None"

    if evidence_reward is None:
        return (
            f"total={reward_info.get('total_score', 0.0):.4f}, "
            f"answer_reward={reward_info.get('answer_reward', 0.0):.4f}"
        )

    answer_term = evidence_reward.answer_weight * float(reward_info.get("answer_reward", 0.0))
    clue_term = evidence_reward.clue_weight * float(reward_info.get("clue_margin_score", 0.0))
    crop_term = evidence_reward.crop_weight * float(reward_info.get("crop_reward", 0.0))
    margin_term = evidence_reward.margin_weight * float(reward_info.get("margin_score", 0.0))

    return (
        f"total={reward_info.get('total_score', 0.0):.4f} "
        f"(answer={answer_term:.4f} + clue={clue_term:.4f} + crop={crop_term:.4f} + margin={margin_term:.4f}), "
        f"answer_score={reward_info.get('answer_score', 0.0):.4f}, "
        f"answer_reward={reward_info.get('answer_reward', 0.0):.4f}, "
        f"answer_margin_delta={reward_info.get('answer_margin_delta', 0.0):.4f}, "
        f"clue_score={reward_info.get('clue_score', 0.0):.4f}, "
        f"clue_margin_delta={reward_info.get('clue_margin_delta', 0.0):.4f}, "
        f"clue_margin={reward_info.get('clue_margin_score', 0.0):.4f}, "
        f"crop_reward={reward_info.get('crop_reward', 0.0):.4f}, "
        f"evidence_gap={reward_info.get('crop_evidence_score', 0.0):.4f}, "
        f"support_gap={reward_info.get('support_gap', 0.0):.4f}, "
        f"mask_drop={reward_info.get('mask_drop', 0.0):.4f}, "
        f"margin={reward_info.get('margin_score', 0.0):.4f}"
    )


def _score_stats(scores):
    if scores is None:
        return None
    scores = scores.detach().float().view(-1)
    if scores.numel() == 0:
        return None

    count = int(scores.numel())
    head_count = max(1, count // 3)
    tail_count = max(1, count // 3)
    topk_count = min(5, count)
    topk_vals = torch.topk(scores, k=topk_count).values
    return {
        "count": count,
        "mean": float(scores.mean().item()),
        "std": float(scores.std(unbiased=False).item()) if count > 1 else 0.0,
        "min": float(scores.min().item()),
        "max": float(scores.max().item()),
        "head_mean": float(scores[:head_count].mean().item()),
        "tail_mean": float(scores[-tail_count:].mean().item()),
        "topk_mean": float(topk_vals.mean().item()),
        "positive_pct": float((scores > 0).float().mean().item() * 100.0),
    }


def _hidden_tensor_to_matrix(hidden_tensor):
    if hidden_tensor is None:
        return None
    hidden = hidden_tensor.detach()
    if hidden.ndim == 3 and hidden.shape[1] == 1:
        hidden = hidden.squeeze(1)
    elif hidden.ndim == 3 and hidden.shape[0] == 1:
        hidden = hidden.squeeze(0)
    elif hidden.ndim == 1:
        hidden = hidden.unsqueeze(0)
    if hidden.ndim != 2:
        return None
    return hidden


def _hidden_list_to_matrix(hidden_states_list):
    if not hidden_states_list:
        return None
    rows = []
    for state in hidden_states_list:
        matrix = _hidden_tensor_to_matrix(state)
        if matrix is None:
            continue
        rows.append(matrix.reshape(-1, matrix.shape[-1]))
    if not rows:
        return None
    return torch.cat(rows, dim=0)


def _candidate_visual_token_ids(model, tokenizer):
    token_ids = []
    config = getattr(model, "config", None)
    for attr in ("image_token_id", "video_token_id"):
        token_id = getattr(config, attr, None) if config is not None else None
        if isinstance(token_id, int) and token_id >= 0:
            token_ids.append(token_id)

    if tokenizer is not None:
        for token in ("<|image_pad|>", "<|video_pad|>", "<IMG_CONTEXT>"):
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
            except Exception:
                token_id = None
            if isinstance(token_id, int) and token_id >= 0:
                token_ids.append(token_id)

    deduped = []
    for token_id in token_ids:
        if token_id not in deduped:
            deduped.append(token_id)
    return deduped


def _compute_visual_prototype(model, tokenizer, inputs):
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        return None, {"available": False, "reason": "missing_input_ids"}

    visual_token_ids = _candidate_visual_token_ids(model, tokenizer)
    if not visual_token_ids:
        return None, {"available": False, "reason": "missing_visual_token_ids"}

    visual_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in visual_token_ids:
        visual_mask |= input_ids == token_id

    visual_token_count = int(visual_mask.sum().item())
    info = {
        "available": False,
        "visual_token_ids": visual_token_ids,
        "visual_token_count": visual_token_count,
    }
    if visual_token_count <= 0:
        info["reason"] = "no_visual_tokens_in_input_ids"
        return None, info

    reset_multimodal_generation_state(model)
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
    )
    last_hidden = outputs.hidden_states[-1]
    visual_mask = visual_mask.to(last_hidden.device)
    visual_hidden = last_hidden[visual_mask]
    if visual_hidden.numel() == 0:
        info["reason"] = "empty_visual_hidden_states"
        return None, info

    prototype = visual_hidden.float().mean(dim=0).detach()
    info.update(
        {
            "available": True,
            "hidden_dim": int(prototype.shape[-1]),
            "prototype_norm": float(prototype.norm().item()),
        }
    )
    return prototype, info


def _visual_focus_block(hidden_matrix, visual_prototype):
    hidden_matrix = _hidden_tensor_to_matrix(hidden_matrix)
    if hidden_matrix is None or visual_prototype is None:
        return None
    hidden = hidden_matrix.float()
    prototype = visual_prototype.to(hidden.device).float().view(1, -1)
    if hidden.shape[-1] != prototype.shape[-1]:
        return None
    cosine = F.cosine_similarity(hidden, prototype.expand_as(hidden), dim=-1)
    focus_score = 0.5 * (cosine + 1.0)
    return {
        "cosine": _score_stats(cosine),
        "score": _score_stats(focus_score),
    }


def _visual_focus_scores(hidden_matrix, visual_prototype):
    hidden_matrix = _hidden_tensor_to_matrix(hidden_matrix)
    if hidden_matrix is None or visual_prototype is None:
        return None
    hidden = hidden_matrix.float()
    prototype = visual_prototype.to(hidden.device).float().view(1, -1)
    if hidden.shape[-1] != prototype.shape[-1]:
        return None
    cosine = F.cosine_similarity(hidden, prototype.expand_as(hidden), dim=-1)
    return 0.5 * (cosine + 1.0)


def _augment_visual_tensor(tensor, drop_prob=0.5):
    """Patch-style visual corruption for already-processed multimodal tensors."""
    if not torch.is_tensor(tensor) or not tensor.is_floating_point() or tensor.numel() == 0:
        return tensor
    drop_prob = max(0.0, min(float(drop_prob), 1.0))
    if drop_prob <= 0.0:
        return tensor.clone()

    augmented = tensor.clone()
    if augmented.ndim >= 2:
        mask_shape = [augmented.shape[0]] + [1] * (augmented.ndim - 1)
        keep_mask = torch.rand(mask_shape, device=augmented.device) >= drop_prob
        return augmented * keep_mask.to(dtype=augmented.dtype)

    keep_mask = torch.rand_like(augmented.float()) >= drop_prob
    return augmented * keep_mask.to(dtype=augmented.dtype)


def _build_augmented_visual_inputs(inputs, drop_prob=0.5):
    augmented = {}
    changed = False
    for key, value in inputs.items():
        if torch.is_tensor(value):
            augmented[key] = value.clone()
        else:
            augmented[key] = value

    for key in ("pixel_values", "pixel_values_videos"):
        if key in augmented and torch.is_tensor(augmented[key]):
            new_value = _augment_visual_tensor(augmented[key], drop_prob=drop_prob)
            changed = changed or new_value.data_ptr() != augmented[key].data_ptr()
            augmented[key] = new_value
    return augmented, changed


def _vppo_random_patch_blackening(pil_img, patch_size=14, black_prob=0.5):
    """VPPO image augmentation: randomly blacken square patches in a raw PIL image."""
    if pil_img is None:
        return None
    if not isinstance(pil_img, Image.Image):
        if hasattr(pil_img, "convert"):
            pil_img = pil_img.convert("RGB")
        else:
            raise TypeError(f"Expected a PIL image, got {type(pil_img)!r}")

    patch_size = max(1, int(patch_size))
    black_prob = max(0.0, min(float(black_prob), 1.0))
    img = np.array(pil_img).astype(np.float32)
    h, w = img.shape[:2]
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            if np.random.rand() < black_prob:
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                if img.ndim == 3:
                    img[y:y_end, x:x_end, :] = 0
                else:
                    img[y:y_end, x:x_end] = 0
    return Image.fromarray(img.astype(np.uint8))


def _get_raw_images_from_media(media):
    if not media:
        return []
    images = media.get("images")
    if images:
        return [image for image in images if image is not None]
    image = media.get("image")
    return [image] if image is not None else []


def _build_vppo_augmented_visual_inputs(
    processor,
    input_text,
    media,
    drop_prob=0.5,
    patch_size=14,
):
    info = {
        "available": False,
        "reason": "not_computed",
        "augmentation_source": "raw_image",
        "patch_size": int(patch_size),
        "drop_prob": float(drop_prob),
    }
    if processor is None:
        info["reason"] = "missing_processor"
        return None, info
    if not input_text:
        info["reason"] = "missing_input_text"
        return None, info

    images = _get_raw_images_from_media(media)
    if not images:
        info["reason"] = "no_raw_images"
        return None, info

    try:
        augmented_images = [
            _vppo_random_patch_blackening(image, patch_size=patch_size, black_prob=drop_prob)
            for image in images
        ]
        augmented_inputs = processor(
            images=augmented_images,
            text=[input_text],
            return_tensors="pt",
        )
    except Exception as exc:
        info["reason"] = f"processor_failed:{type(exc).__name__}"
        info["error"] = str(exc)
        return None, info

    info.update(
        {
            "available": True,
            "reason": "ok",
            "num_images": int(len(augmented_images)),
        }
    )
    return augmented_inputs, info


def _token_logprobs_for_generated_ids(model, inputs, full_input_ids, prompt_length):
    """Teacher-forced log p for generated tokens in full_input_ids."""
    prompt_length = int(prompt_length)
    if full_input_ids is None or full_input_ids.ndim != 2:
        return None
    total_length = int(full_input_ids.shape[-1])
    generated_length = total_length - prompt_length
    if generated_length <= 0 or prompt_length <= 0:
        return None

    attention_mask = torch.ones_like(full_input_ids)
    forward_inputs = {
        "input_ids": full_input_ids,
        "attention_mask": attention_mask,
    }
    forward_inputs.update(
        build_generation_kwargs(
            inputs,
            target_input_ids=full_input_ids,
            include_mm_token_type_ids=True,
        )
    )
    for key in GENERATION_MEDIA_KEYS:
        if key in inputs and key not in forward_inputs:
            forward_inputs[key] = inputs[key]

    reset_multimodal_generation_state(model)
    with torch.no_grad():
        outputs = model(
            **forward_inputs,
            use_cache=False,
            return_dict=True,
        )

    logits = outputs.logits
    if logits is None or logits.shape[-2] < total_length:
        return None
    pred_logits = logits[:, prompt_length - 1 : total_length - 1, :].float()
    target_ids = full_input_ids[:, prompt_length:total_length].to(pred_logits.device)
    log_probs = torch.log_softmax(pred_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs[0].detach()


def _compute_image_sensitivity_scores(
    model,
    inputs,
    full_input_ids,
    prompt_length,
    start_index,
    update_length,
    drop_prob=0.5,
    augmentation_mode="processed_tensor",
    processor=None,
    input_text=None,
    media=None,
    patch_size=14,
):
    """VPPO-style image sensitivity: logprob shift under visual corruption."""
    augmentation_mode = str(augmentation_mode or "processed_tensor")
    info = {
        "available": False,
        "reason": "not_computed",
        "drop_prob": float(drop_prob),
        "augmentation_mode": augmentation_mode,
    }
    if full_input_ids is None or full_input_ids.ndim != 2:
        info["reason"] = "missing_full_input_ids"
        return None, info

    prompt_length = int(prompt_length)
    start_index = int(start_index)
    update_length = int(update_length)
    generated_length = int(full_input_ids.shape[-1]) - prompt_length
    if generated_length <= 0 or update_length <= 0:
        info["reason"] = "empty_generated_or_update_slice"
        return None, info

    if augmentation_mode == "vppo_raw_image":
        augmented_inputs, aug_info = _build_vppo_augmented_visual_inputs(
            processor=processor,
            input_text=input_text,
            media=media,
            drop_prob=drop_prob,
            patch_size=patch_size,
        )
        info.update({f"raw_{key}": value for key, value in aug_info.items()})
        if not aug_info.get("available"):
            info["reason"] = aug_info.get("reason", "raw_image_augmentation_failed")
            return None, info

        augmented_inputs = _move_model_inputs(augmented_inputs, full_input_ids.device)
        augmented_prompt_ids = augmented_inputs.get("input_ids")
        if torch.is_tensor(augmented_prompt_ids):
            aug_prompt_length = int(augmented_prompt_ids.shape[-1])
            info["augmented_prompt_length"] = aug_prompt_length
            info["prompt_length"] = prompt_length
            if aug_prompt_length != prompt_length:
                info["reason"] = "augmented_prompt_length_mismatch"
                return None, info
            prompt_ids_match = bool(torch.equal(augmented_prompt_ids[:, :prompt_length], full_input_ids[:, :prompt_length]))
            info["prompt_ids_match"] = prompt_ids_match
            if not prompt_ids_match:
                info["reason"] = "augmented_prompt_ids_mismatch"
                return None, info
    else:
        augmented_inputs, changed = _build_augmented_visual_inputs(inputs, drop_prob=drop_prob)
        if not changed:
            info["reason"] = "no_visual_tensor_to_augment"
            return None, info

    original_log_probs = _token_logprobs_for_generated_ids(
        model=model,
        inputs=inputs,
        full_input_ids=full_input_ids,
        prompt_length=prompt_length,
    )
    augmented_log_probs = _token_logprobs_for_generated_ids(
        model=model,
        inputs=augmented_inputs,
        full_input_ids=full_input_ids,
        prompt_length=prompt_length,
    )
    if original_log_probs is None or augmented_log_probs is None:
        info["reason"] = "failed_logprob_forward"
        return None, info

    length = min(int(original_log_probs.numel()), int(augmented_log_probs.numel()))
    if length <= 0:
        info["reason"] = "empty_logprob_vectors"
        return None, info

    original_log_probs = original_log_probs[:length]
    augmented_log_probs = augmented_log_probs[:length]
    log_probs_diff = (augmented_log_probs - original_log_probs).clamp(-20.0, 20.0)
    sensitivity_all = torch.exp(log_probs_diff) - log_probs_diff - 1.0
    sensitivity_all = torch.clamp(sensitivity_all, min=0.0, max=10.0).detach().to(full_input_ids.device)

    scores = torch.zeros(update_length, device=full_input_ids.device, dtype=torch.float32)
    valid_mask = torch.zeros(update_length, device=full_input_ids.device, dtype=torch.bool)
    for local_idx in range(update_length):
        generated_idx = start_index + local_idx
        if 0 <= generated_idx < length:
            scores[local_idx] = sensitivity_all[generated_idx].float()
            valid_mask[local_idx] = True

    if not bool(valid_mask.any().item()):
        info["reason"] = "update_slice_outside_generated_ids"
        return None, info

    selected_scores = scores[valid_mask]
    info.update(
        {
            "available": True,
            "reason": "ok",
            "generated_length": int(generated_length),
            "scored_generated_length": int(length),
            "update_length": int(update_length),
            "valid_update_tokens": int(valid_mask.sum().item()),
            "score_stats": _score_stats(selected_scores),
            "logprob_diff_stats": _score_stats(log_probs_diff),
        }
    )
    return scores.detach(), info


def _mean_from_focus_block(block, key="score"):
    if not block:
        return None
    stats = block.get(key) or {}
    mean = stats.get("mean")
    return None if mean is None else float(mean)


def _safe_weighted_mean(values, weights, eps=1e-8):
    values = values.view(-1).float()
    weights = weights.view(-1).float().to(values.device)
    denom = weights.sum()
    if float(denom.detach().abs().item()) <= eps:
        return values.mean()
    return (values * weights).sum() / (denom + eps)


def _span_mask(length, start_ratio=0.0, end_ratio=1.0, device=None):
    length = int(length)
    device = device or torch.device("cpu")
    mask = torch.zeros(length, device=device, dtype=torch.bool)
    if length <= 0:
        return mask
    start_ratio = max(0.0, min(float(start_ratio), 1.0))
    end_ratio = max(0.0, min(float(end_ratio), 1.0))
    if end_ratio < start_ratio:
        start_ratio, end_ratio = end_ratio, start_ratio
    start = max(0, min(length, int(length * start_ratio)))
    end = max(start, min(length, int(length * end_ratio + 0.999999)))
    if end > start:
        mask[start:end] = True
    return mask


def _hard_topk_mask(scores, ratio, candidate_mask=None):
    scores = scores.view(-1).float()
    device = scores.device
    length = scores.numel()
    mask = torch.zeros(length, device=device, dtype=torch.bool)
    ratio = float(ratio)
    if length <= 0 or ratio <= 0:
        return mask
    if candidate_mask is None:
        candidate_mask = torch.ones(length, device=device, dtype=torch.bool)
    else:
        candidate_mask = candidate_mask.view(-1).bool().to(device)
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).view(-1)
    count = int(candidate_indices.numel())
    if count <= 0:
        return mask
    k = max(1, int(count * min(ratio, 1.0) + 0.999999))
    selected_local = torch.topk(scores[candidate_indices], k=min(k, count), largest=True).indices
    mask[candidate_indices[selected_local]] = True
    return mask


def _uniform_topk_mask(length, ratio, candidate_mask=None, device=None):
    length = int(length)
    if candidate_mask is not None:
        device = candidate_mask.device
    device = device or torch.device("cpu")
    mask = torch.zeros(length, device=device, dtype=torch.bool)
    ratio = float(ratio)
    if length <= 0 or ratio <= 0:
        return mask
    if candidate_mask is None:
        candidate_indices = torch.arange(length, device=device)
    else:
        candidate_indices = torch.nonzero(candidate_mask.view(-1).bool().to(device), as_tuple=False).view(-1)
    count = int(candidate_indices.numel())
    if count <= 0:
        return mask
    k = max(1, int(count * min(ratio, 1.0) + 0.999999))
    if k >= count:
        selected = candidate_indices
    else:
        positions = torch.linspace(0, count - 1, steps=k, device=device).round().long()
        positions = torch.unique(positions, sorted=True)
        if int(positions.numel()) < k:
            fill = torch.arange(count, device=device)
            positions = torch.unique(torch.cat([positions, fill]), sorted=True)
        selected = candidate_indices[positions[:k]]
    mask[selected] = True
    return mask


def _centered_evidence_scores(reward_info, evidence_reward=None, fallback_score=0.0, reward_center=0.0):
    if reward_info is None or evidence_reward is None:
        centered = float(fallback_score) - float(reward_center)
        return {
            "reasoning_score": centered,
            "visual_score": 0.0,
            "centered_total_score": centered,
            "answer_term": centered,
            "margin_term": 0.0,
            "clue_term": 0.0,
            "crop_term": 0.0,
        }

    answer_reward = float(reward_info.get("answer_reward", 0.0))
    margin_score = float(reward_info.get("margin_score", 0.0))
    clue_margin_score = float(reward_info.get("clue_margin_score", 0.0))
    crop_reward = float(reward_info.get("crop_reward", 0.0))
    answer_center = float(getattr(evidence_reward, "answer_reward_center", 0.0))

    answer_term = evidence_reward.answer_weight * (answer_reward - answer_center)
    margin_term = evidence_reward.margin_weight * margin_score
    clue_term = evidence_reward.clue_weight * clue_margin_score
    crop_term = evidence_reward.crop_weight * crop_reward
    reasoning_score = answer_term + margin_term
    visual_score = clue_term + crop_term

    return {
        "reasoning_score": float(reasoning_score),
        "visual_score": float(visual_score),
        "centered_total_score": float(reasoning_score + visual_score),
        "answer_term": float(answer_term),
        "margin_term": float(margin_term),
        "clue_term": float(clue_term),
        "crop_term": float(crop_term),
    }


def _uses_visual_engagement_reward(mode):
    return str(mode or "ecl").strip().lower() in {
        "aver",
        "visual_engagement",
        "attention_engagement",
    }


def _uses_answer_likelihood_delta_visual_reward(mode):
    return str(mode or "ecl").strip().lower() in {
        "answer_likelihood_delta",
        "image_likelihood_delta",
    }


def _cap_generated_ids_for_visual_engagement(generated_ids, max_score_tokens=128):
    generated_ids = list(generated_ids or [])
    max_score_tokens = int(max_score_tokens or 0)
    if max_score_tokens > 0:
        return generated_ids[:max_score_tokens]
    return generated_ids


def _compute_token_router_weights(
    optimized_hidden_states,
    token_entropy,
    visual_prototype=None,
    image_sensitivity_scores=None,
    reason_router_scores=None,
    reason_router_signal="entropy",
    token_router="visual_focus_entropy",
    visual_threshold=0.5,
    entropy_threshold=0.0,
    router_temperature=0.5,
    visual_topk_ratio=0.25,
    reason_topk_ratio=0.25,
    visual_span_start_ratio=0.0,
    visual_span_end_ratio=1.0,
    reason_span_start_ratio=0.5,
    reason_span_end_ratio=1.0,
):
    update_length = optimized_hidden_states.shape[0]
    device = optimized_hidden_states.device
    router_temperature = max(float(router_temperature), 1e-6)

    entropy_detached = token_entropy.detach().view(-1).float().to(device)
    if entropy_detached.numel() > 1:
        entropy_std = entropy_detached.std(unbiased=False)
        entropy_norm = (entropy_detached - entropy_detached.mean()) / (entropy_std + 1e-6)
    else:
        entropy_norm = torch.zeros_like(entropy_detached)

    if reason_router_scores is None:
        reason_scores = entropy_detached
        reason_scores_norm = entropy_norm
        reason_router_signal = "entropy"
    else:
        reason_scores = reason_router_scores.detach().view(-1).float().to(device)
        if reason_scores.numel() != update_length:
            fixed = torch.zeros(update_length, device=device, dtype=torch.float32)
            copy_length = min(update_length, int(reason_scores.numel()))
            if copy_length > 0:
                fixed[:copy_length] = reason_scores[:copy_length]
            reason_scores = fixed
        if reason_scores.numel() > 1:
            reason_std = reason_scores.std(unbiased=False)
            reason_scores_norm = (reason_scores - reason_scores.mean()) / (reason_std + 1e-6)
        else:
            reason_scores_norm = torch.zeros_like(reason_scores)

    visual_focus = _visual_focus_scores(optimized_hidden_states.detach(), visual_prototype)
    visual_available = visual_focus is not None
    if visual_focus is None:
        visual_focus = torch.zeros(update_length, device=device, dtype=torch.float32)
    else:
        visual_focus = visual_focus.detach().view(-1).float().to(device)

    image_sensitivity_available = image_sensitivity_scores is not None
    if image_sensitivity_scores is None:
        image_sensitivity = torch.zeros(update_length, device=device, dtype=torch.float32)
    else:
        image_sensitivity = image_sensitivity_scores.detach().view(-1).float().to(device)
        if image_sensitivity.numel() != update_length:
            fixed = torch.zeros(update_length, device=device, dtype=torch.float32)
            copy_length = min(update_length, int(image_sensitivity.numel()))
            if copy_length > 0:
                fixed[:copy_length] = image_sensitivity[:copy_length]
            image_sensitivity = fixed

    visual_span = _span_mask(update_length, visual_span_start_ratio, visual_span_end_ratio, device=device)
    reason_span = _span_mask(update_length, reason_span_start_ratio, reason_span_end_ratio, device=device)

    if token_router == "hard_topk":
        visual_mask = (
            _hard_topk_mask(visual_focus, visual_topk_ratio)
            if visual_available
            else torch.zeros(update_length, device=device, dtype=torch.bool)
        )
        reason_mask = _hard_topk_mask(reason_scores_norm, reason_topk_ratio, candidate_mask=~visual_mask)
        w_visual = visual_mask.float()
        w_reason = reason_mask.float()
    elif token_router in OVERLAP_IMAGE_SENSITIVITY_ROUTERS:
        visual_mask = (
            _hard_topk_mask(image_sensitivity, visual_topk_ratio)
            if image_sensitivity_available
            else torch.zeros(update_length, device=device, dtype=torch.bool)
        )
        reason_mask = _hard_topk_mask(entropy_norm, reason_topk_ratio)
        w_visual = visual_mask.float()
        w_reason = reason_mask.float()
    elif token_router in SWAPPED_IMAGE_SENSITIVITY_ROUTERS:
        visual_mask = _hard_topk_mask(entropy_norm, visual_topk_ratio)
        reason_mask = (
            _hard_topk_mask(image_sensitivity, reason_topk_ratio, candidate_mask=~visual_mask)
            if image_sensitivity_available
            else torch.zeros(update_length, device=device, dtype=torch.bool)
        )
        w_visual = visual_mask.float()
        w_reason = reason_mask.float()
    elif token_router in IMAGE_SENSITIVITY_ROUTERS:
        visual_mask = (
            _hard_topk_mask(image_sensitivity, visual_topk_ratio)
            if image_sensitivity_available
            else torch.zeros(update_length, device=device, dtype=torch.bool)
        )
        reason_mask = _hard_topk_mask(reason_scores_norm, reason_topk_ratio, candidate_mask=~visual_mask)
        w_visual = visual_mask.float()
        w_reason = reason_mask.float()
    elif token_router == "random_topk":
        visual_mask = _hard_topk_mask(
            torch.rand(update_length, device=device, dtype=torch.float32),
            visual_topk_ratio,
        )
        reason_mask = _hard_topk_mask(
            torch.rand(update_length, device=device, dtype=torch.float32),
            reason_topk_ratio,
            candidate_mask=~visual_mask,
        )
        w_visual = visual_mask.float()
        w_reason = reason_mask.float()
    elif token_router == "uniform_topk":
        visual_mask = _uniform_topk_mask(update_length, visual_topk_ratio, device=device)
        reason_mask = _uniform_topk_mask(update_length, reason_topk_ratio, candidate_mask=~visual_mask)
        w_visual = visual_mask.float()
        w_reason = reason_mask.float()
    elif token_router == "span_constrained":
        w_visual = torch.sigmoid((visual_focus - float(visual_threshold)) / router_temperature)
        if not visual_available:
            w_visual = torch.zeros_like(w_visual)
        w_visual = w_visual * visual_span.float()
        entropy_gate = torch.sigmoid((entropy_norm - float(entropy_threshold)) / router_temperature)
        w_reason = entropy_gate * (1.0 - w_visual) * reason_span.float()
    elif token_router == "hard_topk_span":
        visual_mask = (
            _hard_topk_mask(visual_focus, visual_topk_ratio, candidate_mask=visual_span)
            if visual_available
            else torch.zeros(update_length, device=device, dtype=torch.bool)
        )
        reason_candidates = reason_span & (~visual_mask)
        reason_mask = _hard_topk_mask(reason_scores_norm, reason_topk_ratio, candidate_mask=reason_candidates)
        w_visual = visual_mask.float()
        w_reason = reason_mask.float()
    elif token_router == "entropy_only":
        w_visual = torch.zeros_like(entropy_norm)
        w_reason = torch.sigmoid((reason_scores_norm - float(entropy_threshold)) / router_temperature)
    elif token_router == "visual_focus_only":
        w_visual = torch.sigmoid((visual_focus - float(visual_threshold)) / router_temperature)
        if not visual_available:
            w_visual = torch.zeros_like(w_visual)
        w_reason = 1.0 - w_visual
    else:
        w_visual = torch.sigmoid((visual_focus - float(visual_threshold)) / router_temperature)
        if not visual_available:
            w_visual = torch.zeros_like(w_visual)
        entropy_gate = torch.sigmoid((reason_scores_norm - float(entropy_threshold)) / router_temperature)
        w_reason = entropy_gate * (1.0 - w_visual)

    if float(w_reason.sum().detach().item()) <= 1e-8 and float(w_visual.sum().detach().item()) <= 1e-8:
        w_reason = torch.ones_like(entropy_norm)

    return {
        "w_visual": w_visual.detach(),
        "w_reason": w_reason.detach(),
        "visual_focus": visual_focus.detach(),
        "entropy_norm": entropy_norm.detach(),
        "reason_scores": reason_scores.detach(),
        "reason_scores_norm": reason_scores_norm.detach(),
        "visual_available": bool(visual_available),
        "stats": {
            "visual_focus": _score_stats(visual_focus),
            "image_sensitivity": _score_stats(image_sensitivity),
            "entropy": _score_stats(entropy_detached),
            "entropy_norm": _score_stats(entropy_norm),
            "reason_scores": _score_stats(reason_scores),
            "reason_scores_norm": _score_stats(reason_scores_norm),
            "w_visual": _score_stats(w_visual),
            "w_reason": _score_stats(w_reason),
            "visual_span": _score_stats(visual_span.float()),
            "reason_span": _score_stats(reason_span.float()),
            "visual_available": bool(visual_available),
            "image_sensitivity_available": bool(image_sensitivity_available),
            "token_router": token_router,
            "reason_router_signal": str(reason_router_signal),
            "visual_topk_ratio": float(visual_topk_ratio),
            "reason_topk_ratio": float(reason_topk_ratio),
            "visual_span_start_ratio": float(visual_span_start_ratio),
            "visual_span_end_ratio": float(visual_span_end_ratio),
            "reason_span_start_ratio": float(reason_span_start_ratio),
            "reason_span_end_ratio": float(reason_span_end_ratio),
        },
    }


def _conditioned_clue_answer_key(answer_text, evidence_reward=None, problem=None):
    if evidence_reward is not None:
        try:
            key = evidence_reward._normalize_answer_display(answer_text, problem=problem)
            if key:
                return str(key)
        except Exception:
            pass
    short = _extract_short_answer(answer_text, problem=problem)
    if short:
        return str(short).strip()
    return _preview_text(answer_text, limit=160)


def _get_candidate_conditioned_clues(
    cache,
    model,
    tokenizer,
    processor,
    device,
    problem,
    media,
    answer_key,
    max_new_tokens,
    num_clues,
):
    if answer_key in cache:
        return cache[answer_key]

    clues = generate_candidate_visual_clues(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        device=device,
        problem=problem,
        media=media,
        candidate_answer=answer_key,
        max_new_tokens=max_new_tokens,
        num_clues=num_clues,
    )
    cache[answer_key] = list(clues or [])
    return cache[answer_key]


def optimized_generation(
        reward_model, model, tokenizer, device,
        question, input_text, original_answer,
        original_hidden_states_list, input_ids, start_index=0,
        prompt_inputs=None, media=None, problem=None,
        max_num_steps=10, lr=0.03, max_new_tokens=1024,
        grad_clip=None, k=0.1, reward_threshold=-0.2, latent_reg_lambda=0.0,
        update_position_strategy="auto"):
    '''
    Generate answer using optimized generation process

    Args:
        reward_model: reward model
        model: language model
        tokenizer: tokenizer
        device: device to use
        question: question
        input_text: formatted prompt
        original_answer: original generated answer
        original_hidden_states_list: list of hidden states for each token
        input_ids: the input_ids of original generation
        start_index: the start index of the optimized hidden states
        max_num_steps: number of optimization steps
        lr: learning rate
        max_new_tokens: maximum number of new tokens to generate
        grad_clip: gradient clipping threshold
        k: ratio of update length to the total length of hidden states
        reward_threshold: threshold for the reward to stop optimization

    Returns:
        final_answer: the final generated answer
        reward_history: list of rewards during optimization
        original_length: length of the original answer
        optimized_length: length of the optimized answer
        update_length: length of the optimized hidden states
    '''
    eos_token = tokenizer.eos_token
    if eos_token is not None and eos_token not in stop_words:
        stop_words.append(eos_token)
    reward_history = []
    initial_reward = reward_model.get_reward(question, original_answer, media=media, problem=problem)

    print(f"-- Original Output: {original_answer} -- Initial Reward: {initial_reward}")
    reward_history.append(initial_reward)
    current_reward = initial_reward

    original_length = len(original_hidden_states_list)
    optimized_length = 0

    if max_num_steps <= 0:
        optimized_length = len(tokenizer.encode(original_answer))
        return original_answer, reward_history, original_length, optimized_length, 0

    if prompt_inputs is None:
        inputs = tokenizer([input_text], return_tensors="pt").to(device)
    else:
        inputs = move_model_inputs(prompt_inputs, device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    stop_token_ids = get_stop_token_ids(tokenizer)
    base_input_ids = inputs["input_ids"].clone()
    prompt_length = base_input_ids.shape[-1]

    # grab update fraction
    update_length = min(int(k * original_length), 300)
    if update_length <= 0:
        print("Update Length Zero!!!")
        final_answer = original_answer
        return final_answer, reward_history, original_length, optimized_length, update_length

    if update_position_strategy == "auto":
        update_position_strategy = "tail" if (problem is not None or media is not None) else "head"
    if update_position_strategy == "tail":
        start_index = max(0, original_length - update_length)
    else:
        start_index = min(max(0, start_index), max(original_length - update_length, 0))

    anchor_hidden_states = torch.stack(
        [
            state.clone().detach()
            for state in original_hidden_states_list[start_index: min(start_index + update_length, len(original_hidden_states_list))]
        ]
    )
    optimized_hidden_states = torch.nn.Parameter(anchor_hidden_states.clone())

    # configure optimizer
    optimizer = torch.optim.Adam([optimized_hidden_states], lr=lr)

    input_ids = input_ids[:, : base_input_ids.shape[-1] + start_index]
    base_input_ids = input_ids.clone()
    new_answer = None

    # optimization loop
    for _ in range(max_num_steps):
        input_ids = base_input_ids.clone()
        if current_reward > reward_threshold:
            final_answer = new_answer if new_answer is not None else original_answer
            optimized_length = len(tokenizer.encode(final_answer))
            print(f"-- Final Answer: {final_answer}, -- Current Reward: {current_reward}")
            return final_answer, reward_history, original_length, optimized_length, update_length

        optimizer.zero_grad()

        logits = model.lm_head(optimized_hidden_states) #[update_length, 1, vocab_size]
        probs = torch.softmax(logits, dim=-1) + 1e-8    #[update_length, 1, vocab_size]

        next_token_ids = torch.argmax(probs, dim=-1)    #[update_length, 1]
        next_token_ids = next_token_ids.squeeze(-1)    #[update_length]
        log_pi_xz = torch.log(
            probs[torch.arange(update_length, device=optimized_hidden_states.device), 0, next_token_ids] + 1e-10
        )

        # total loss
        loss = - current_reward * log_pi_xz.sum()
        if latent_reg_lambda > 0:
            latent_reg = torch.mean((optimized_hidden_states - anchor_hidden_states) ** 2)
            loss = loss + latent_reg_lambda * latent_reg
        print(f"-- Loss: {loss.item()}")
        loss.backward(retain_graph=True)

        if grad_clip:
            torch.nn.utils.clip_grad_norm_([optimized_hidden_states], grad_clip)
        optimizer.step()

        with torch.no_grad():
            next_tokens = torch.argmax(model.lm_head(optimized_hidden_states),
                                       dim=-1) #[update_length, 1]
            next_tokens = next_tokens.squeeze(-1) #[update_length]
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(0)], dim=-1)

        # generate full answer
        with torch.no_grad():
            reset_multimodal_generation_state(model)
            generation_inputs = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
            generation_inputs.update(
                build_generation_kwargs(
                    inputs,
                    target_input_ids=input_ids,
                    include_mm_token_type_ids=True,
                )
            )
            generated_sequences = model.generate(
                **generation_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
            )

        del next_tokens, input_ids
        torch.cuda.empty_cache()

        new_answer = tokenizer.decode(
            generated_sequences[0][prompt_length:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        new_answer = trim_after_stop_words(new_answer, stop_words)
        current_reward = reward_model.get_reward(question, new_answer, media=media, problem=problem)
        print(f"-- New Answer: {new_answer}, -- Current Reward: {current_reward}")

        reward_history.append(current_reward)

    final_answer = new_answer
    optimized_length = len(tokenizer.encode(final_answer))
    print(f"-- Final answer: {final_answer}")
    return final_answer, reward_history, original_length, optimized_length, update_length


# ---- LatentGrad v2 ----

"""
LatentGrad: Dual-space test-time optimization for multimodal reasoning.
"""

from collections import Counter

import torch
import torch.nn.functional as F
from methods.latentgrad.generation_utils import (
    build_generation_kwargs,
    get_stop_token_ids,
    reset_multimodal_generation_state,
    trim_after_stop_words,
)

try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs):
        return text


stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]


def _move_model_inputs(model_inputs, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in model_inputs.items()
    }


def _module_device(module, fallback=None):
    if module is None:
        return fallback
    try:
        return next(module.parameters()).device
    except StopIteration:
        return fallback
    except Exception:
        return fallback


def _generate_from_hidden_prefix(
    model,
    tokenizer,
    prompt_length,
    base_input_ids_trimmed,
    prefix_token_ids,
    inputs,
    max_new_tokens,
    return_trace=False,
):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prefix_token_ids = prefix_token_ids.to(base_input_ids_trimmed.device)
    input_ids = torch.cat([base_input_ids_trimmed, prefix_token_ids.unsqueeze(0)], dim=-1)
    generation_inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    generation_inputs.update(
        build_generation_kwargs(
            inputs,
            target_input_ids=input_ids,
            include_mm_token_type_ids=True,
        )
    )

    reset_multimodal_generation_state(model)
    with torch.no_grad():
        generated_sequences = model.generate(
            **generation_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=get_stop_token_ids(tokenizer),
        )

    answer = tokenizer.decode(
        generated_sequences[0][prompt_length:],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    answer = trim_after_stop_words(answer, stop_words)
    if not return_trace:
        return answer

    generated_ids_all = generated_sequences[0][prompt_length:].detach().cpu().tolist()
    return {
        "text": answer,
        "generated_ids": strip_stop_tokens(generated_ids_all, tokenizer),
        "generated_ids_all": generated_ids_all,
    }


def _extract_short_answer(text, problem=None):
    import json
    import re

    if text is None:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            for key in ["final answer", "final_answer", "answer"]:
                if key in obj:
                    val = str(obj[key]).strip()
                    if val:
                        return val
        except Exception:
            pass

    for pattern in [
        r'"final answer"\s*:\s*"?(.*?)"?\s*[,}\n]',
        r'"final_answer"\s*:\s*"?(.*?)"?\s*[,}\n]',
        r"(?:final answer)\s*[:：]\s*(.+)",
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip().strip('"').strip("'")
    return None


def _select_by_consistency_and_score(candidates, mode="majority", score_temperature=1.0):
    if not candidates:
        return None, None

    answer_groups = {}
    for full_text, short_ans, score in candidates:
        key = str(short_ans).strip().lower() if short_ans else "__none__"
        if key not in answer_groups:
            answer_groups[key] = []
        answer_groups[key].append((full_text, short_ans, score))

    best_key = None
    best_score = float("-inf")
    score_temperature = max(float(score_temperature), 1e-6)
    for key, group in answer_groups.items():
        scores = torch.tensor([s for _, _, s in group], dtype=torch.float32)
        if mode == "reward_weighted":
            composite = torch.logsumexp(scores / score_temperature, dim=0).item()
        else:
            count = len(group)
            max_score = max(s for _, _, s in group)
            composite = count * 10.0 + max_score
        if composite > best_score:
            best_score = composite
            best_key = key

    best_group = answer_groups[best_key]
    best_group.sort(key=lambda x: x[2], reverse=True)
    return best_group[0][0], best_group[0][1]


def optimized_generation_v2(
    confidence_reward,
    reward_model,
    model,
    tokenizer,
    device,
    question,
    input_text,
    original_answer,
    original_hidden_states_list,
    input_ids,
    start_index=0,
    prompt_inputs=None,
    processor=None,
    media=None,
    problem=None,
    max_num_steps=10,
    lr=0.03,
    max_new_tokens=1024,
    grad_clip=None,
    k=0.1,
    reward_threshold=-0.2,
    anchor_weight=0.02,
    logit_lr=0.01,
    noise_std=0.3,
    noise_candidates=3,
    dual_space=True,
    use_confidence_reward=True,
    confidence_reward_mix=0.7,
    visual_reward=None,
    evidence_reward=None,
    visual_clues=None,
    crop_evidence=None,
    candidate_aggregation="majority",
    aggregation_temperature=0.5,
    debug_loss_terms=False,
    debug_reward_terms=False,
    debug_candidate_rewards=False,
    debug_visual_focus=False,
    debug_candidate_conditioned_clues=False,
    candidate_conditioned_clue_top_k=2,
    candidate_conditioned_clue_max_new_tokens=64,
    use_token_disentangle=False,
    token_router="visual_focus_entropy",
    reason_router_signal="entropy",
    image_sensitivity_drop_prob=0.5,
    image_sensitivity_patch_size=14,
    visual_router_threshold=0.5,
    entropy_router_threshold=0.0,
    router_temperature=0.5,
    visual_topk_ratio=0.25,
    reason_topk_ratio=0.25,
    visual_span_start_ratio=0.0,
    visual_span_end_ratio=1.0,
    reason_span_start_ratio=0.5,
    reason_span_end_ratio=1.0,
    visual_policy_weight=1.0,
    reason_policy_weight=1.0,
    visual_anchor_extra=0.0,
    visual_branch_reward_mode="ecl",
    visual_engagement_tau=0.2,
    visual_engagement_layer_index=-1,
    visual_engagement_max_score_tokens=128,
    visual_engagement_model=None,
    visual_engagement_tokenizer=None,
    visual_engagement_device=None,
    reward_diagnostics=None,
):
    eos_token = tokenizer.eos_token
    if eos_token is not None and eos_token not in stop_words:
        stop_words.append(eos_token)

    reward_history = []
    reward_history_detailed = []
    loss_history_detailed = []
    candidate_reward_history = []
    candidate_conditioned_clue_history = []
    debug_reward_terms = bool(debug_reward_terms or debug_candidate_rewards)
    candidate_conditioned_clue_cache = {}
    reference_conditioned_answer_key = None
    reference_conditioned_clues = []
    reference_conditioned_info = None
    candidate_conditioned_clue_processor = None if visual_reward is None else getattr(visual_reward, "processor", None)
    debug_candidate_conditioned_clues = bool(
        debug_candidate_conditioned_clues
        and evidence_reward is not None
        and media is not None
        and candidate_conditioned_clue_processor is not None
    )
    candidate_conditioned_clue_top_k = max(int(candidate_conditioned_clue_top_k), 0)
    use_visual_engagement_reward = _uses_visual_engagement_reward(visual_branch_reward_mode)
    use_answer_likelihood_delta_visual_reward = _uses_answer_likelihood_delta_visual_reward(
        visual_branch_reward_mode
    )
    visual_engagement_baseline = None
    visual_engagement_baseline_info = None
    visual_engagement_history = []
    if prompt_inputs is None:
        inputs = tokenizer([input_text], return_tensors="pt").to(device)
    else:
        inputs = _move_model_inputs(prompt_inputs, device)
    base_input_ids = inputs["input_ids"].clone()
    prompt_length = base_input_ids.shape[-1]

    visual_engagement_model = visual_engagement_model or model
    visual_engagement_tokenizer = visual_engagement_tokenizer or tokenizer
    if visual_engagement_device is None:
        visual_engagement_device = device
    if use_visual_engagement_reward:
        if prompt_inputs is None:
            visual_engagement_inputs = visual_engagement_tokenizer([input_text], return_tensors="pt").to(visual_engagement_device)
        else:
            visual_engagement_inputs = _move_model_inputs(prompt_inputs, visual_engagement_device)
    else:
        visual_engagement_inputs = None

    if use_visual_engagement_reward:
        original_generated_ids = input_ids[0, prompt_length:].detach().cpu().tolist()
        original_generated_ids = _cap_generated_ids_for_visual_engagement(
            original_generated_ids,
            max_score_tokens=visual_engagement_max_score_tokens,
        )
        visual_engagement_baseline_info = visual_engagement_score(
            model=visual_engagement_model,
            tokenizer=visual_engagement_tokenizer,
            inputs=visual_engagement_inputs,
            generated_ids=original_generated_ids,
            layer_index=visual_engagement_layer_index,
        )
        visual_engagement_baseline = float(visual_engagement_baseline_info["engagement"])
        print(
            colored(
                f"-- AVER baseline: E0={visual_engagement_baseline:.6f}, "
                f"tokens={visual_engagement_baseline_info.get('num_generated_tokens')}, "
                f"image_tokens={visual_engagement_baseline_info.get('num_image_tokens')}, "
                f"tau={float(visual_engagement_tau):.4f}, "
                f"layer={visual_engagement_layer_index}",
                "yellow",
            )
        )

    orig_short = _extract_short_answer(original_answer, problem)
    orig_visual_score = 0.0
    if visual_reward is not None and media is not None:
        orig_visual_score = visual_reward.get_reward(question, original_answer, media=media, problem=problem)

    orig_reward_info = None
    orig_total_score = orig_visual_score
    if evidence_reward is not None:
        orig_infos = evidence_reward.score_candidates(
            question=question,
            answer_texts=[original_answer],
            media=media,
            problem=problem,
            clues=visual_clues,
            crop_evidence=crop_evidence,
            reference_answer=None,
        )
        if orig_infos:
            orig_reward_info = orig_infos[0]
            orig_total_score = orig_reward_info["total_score"]

    print(f"-- Original Output: {original_answer}")
    if orig_reward_info is not None:
        print(
            f"-- Original short: {orig_short}, total_score: {orig_total_score:.4f}, "
            f"answer_reward: {orig_reward_info['answer_reward']:.4f}, "
            f"answer_margin_delta: {orig_reward_info.get('answer_margin_delta', 0.0):.4f}, "
            f"clue_reward: {orig_reward_info['clue_reward']:.4f}, "
            f"clue_margin_delta: {orig_reward_info.get('clue_margin_delta', 0.0):.4f}, "
            f"clue_margin: {orig_reward_info.get('clue_margin_score', 0.0):.4f}, "
            f"crop_reward: {orig_reward_info['crop_reward']:.4f}, "
            f"support_gap: {orig_reward_info.get('support_gap', 0.0):.4f}, "
            f"mask_drop: {orig_reward_info.get('mask_drop', 0.0):.4f}, "
            f"margin: {orig_reward_info['margin_score']:.4f}"
        )
        if debug_reward_terms:
            print(
                colored(
                    "-- Original reward breakdown: "
                    + _format_evidence_reward_breakdown(orig_reward_info, evidence_reward=evidence_reward),
                    "cyan",
                )
            )
    else:
        print(f"-- Original short: {orig_short}, visual_score: {orig_visual_score:.4f}")

    if debug_candidate_conditioned_clues:
        reference_key = _conditioned_clue_answer_key(original_answer, evidence_reward=evidence_reward, problem=problem)
        reference_conditioned_answer_key = reference_key
        reference_conditioned_clues = _get_candidate_conditioned_clues(
            cache=candidate_conditioned_clue_cache,
            model=model,
            tokenizer=tokenizer,
            processor=candidate_conditioned_clue_processor,
            device=device,
            problem=problem,
            media=media,
            answer_key=reference_key,
            max_new_tokens=candidate_conditioned_clue_max_new_tokens,
            num_clues=max(1, len(visual_clues or []) or 3),
        )
        reference_infos = evidence_reward.score_candidate_conditioned_clues(
            answer_texts=[reference_key],
            candidate_clues=[reference_conditioned_clues],
            media=media,
            problem=problem,
            reference_answer=None,
            reference_clues=None,
        )
        reference_conditioned_info = reference_infos[0] if reference_infos else None
        print(
            colored(
                f"-- Candidate-conditioned reference clues: answer={reference_key}, "
                f"num_clues={len(reference_conditioned_clues)}, "
                f"score={0.0 if reference_conditioned_info is None else reference_conditioned_info.get('clue_score', 0.0):.4f}",
                "yellow",
            )
        )

    all_candidates = [(original_answer, orig_short, orig_total_score)]
    reward_history.append(orig_total_score)
    orig_split_scores = _centered_evidence_scores(
        orig_reward_info,
        evidence_reward=evidence_reward,
        fallback_score=orig_total_score,
        reward_center=0.0 if evidence_reward is None else evidence_reward.reward_center,
    )
    if use_visual_engagement_reward:
        orig_split_scores["visual_score"] = 0.0
        orig_split_scores["centered_total_score"] = float(orig_split_scores["reasoning_score"])
        orig_split_scores["clue_term"] = 0.0
        orig_split_scores["crop_term"] = 0.0
    reward_history_detailed.append(
        {
            "step": -1,
            "best_total_score": float(orig_total_score),
            "best_centered_total_score": float(orig_split_scores["centered_total_score"]),
            "best_reasoning_score": float(orig_split_scores["reasoning_score"]),
            "best_visual_score": float(orig_split_scores["visual_score"]),
            "best_answer_term": float(orig_split_scores["answer_term"]),
            "best_margin_term": float(orig_split_scores["margin_term"]),
            "best_clue_term": float(orig_split_scores["clue_term"]),
            "best_crop_term": float(orig_split_scores["crop_term"]),
            "best_answer_score": None if orig_reward_info is None else float(orig_reward_info["answer_score"]),
            "best_answer_reward": None if orig_reward_info is None else float(orig_reward_info["answer_reward"]),
            "best_clue_score": None if orig_reward_info is None else float(orig_reward_info["clue_score"]),
            "best_clue_reward": None if orig_reward_info is None else float(orig_reward_info["clue_reward"]),
            "best_answer_margin_delta": None if orig_reward_info is None else float(orig_reward_info.get("answer_margin_delta", 0.0)),
            "best_clue_margin_delta": None if orig_reward_info is None else float(orig_reward_info.get("clue_margin_delta", 0.0)),
            "best_clue_margin_score": None if orig_reward_info is None else float(orig_reward_info.get("clue_margin_score", 0.0)),
            "best_crop_evidence_score": None if orig_reward_info is None else float(orig_reward_info["crop_evidence_score"]),
            "best_crop_reward": None if orig_reward_info is None else float(orig_reward_info["crop_reward"]),
            "best_support_gap": None if orig_reward_info is None else float(orig_reward_info.get("support_gap", 0.0)),
            "best_mask_drop": None if orig_reward_info is None else float(orig_reward_info.get("mask_drop", 0.0)),
            "best_margin_score": None if orig_reward_info is None else float(orig_reward_info["margin_score"]),
            "best_visual_engagement": None if visual_engagement_baseline_info is None else float(visual_engagement_baseline_info["engagement"]),
            "best_visual_engagement_delta": 0.0 if use_visual_engagement_reward else None,
            "best_visual_engagement_reward": 0.0 if use_visual_engagement_reward else None,
        }
    )

    original_length = len(original_hidden_states_list)
    optimized_length = 0

    if prompt_inputs is None:
        inputs = tokenizer([input_text], return_tensors="pt").to(device)
    else:
        inputs = _move_model_inputs(prompt_inputs, device)
    base_input_ids = inputs["input_ids"].clone()

    start_index = max(0, min(start_index, max(original_length - 1, 0)))
    available_length = max(0, original_length - start_index)
    if available_length > 0:
        update_length = min(max(1, int(k * available_length)), available_length, 300)
    else:
        update_length = 0

    if update_length <= 0:
        print("Update Length Zero!!!")
        return original_answer, reward_history, original_length, optimized_length, update_length

    latent_device = _module_device(getattr(model, "lm_head", None), fallback=torch.device(device))
    original_hidden_slice = torch.stack(
        [
            state.clone().detach()
            for state in original_hidden_states_list[start_index : min(start_index + update_length, len(original_hidden_states_list))]
        ]
    ).to(latent_device)
    optimized_hidden_states = torch.nn.Parameter(original_hidden_slice.clone())
    optimizer = torch.optim.Adam([optimized_hidden_states], lr=lr)

    visual_focus_diagnostics = None
    visual_prototype = None
    prototype_info = {"available": False, "reason": "not_requested"}
    original_update_focus_mean = None
    if debug_visual_focus or use_token_disentangle:
        visual_prototype, prototype_info = _compute_visual_prototype(model, tokenizer, inputs)
    if debug_visual_focus:
        original_all_matrix = _hidden_list_to_matrix(original_hidden_states_list)
        original_all_focus = _visual_focus_block(original_all_matrix, visual_prototype)
        original_update_focus = _visual_focus_block(original_hidden_slice, visual_prototype)
        original_update_focus_mean = _mean_from_focus_block(original_update_focus)
        visual_focus_diagnostics = {
            "enabled": True,
            "prototype": prototype_info,
            "original_all": original_all_focus,
            "original_update_slice": original_update_focus,
            "steps": [],
            "note": "Diagnostics only; visual_focus_score is not included in the ECL-TTS reward.",
            "formula": "visual_focus_score = 0.5 * (cosine(hidden_state, mean(image_token_hidden_states)) + 1)",
        }
        if prototype_info.get("available"):
            all_mean = _mean_from_focus_block(original_all_focus)
            update_mean = _mean_from_focus_block(original_update_focus)
            print(
                colored(
                    f"-- Visual focus original: all_mean={all_mean if all_mean is not None else float('nan'):.4f}, "
                    f"update_mean={update_mean if update_mean is not None else float('nan'):.4f}, "
                    f"visual_tokens={prototype_info.get('visual_token_count')}",
                    "yellow",
                )
            )
        else:
            print(
                colored(
                    f"-- Visual focus unavailable: {prototype_info.get('reason', 'unknown')}",
                    "yellow",
                )
            )
    elif use_token_disentangle:
        if prototype_info.get("available"):
            print(
                colored(
                    f"-- Token router visual prototype: available, visual_tokens={prototype_info.get('visual_token_count')}",
                    "yellow",
                )
            )
        else:
            print(
                colored(
                    f"-- Token router visual prototype unavailable: {prototype_info.get('reason', 'unknown')}",
                    "yellow",
                )
            )

    input_ids_trimmed = input_ids[:, : base_input_ids.shape[-1] + start_index]
    base_input_ids_trimmed = input_ids_trimmed.clone()
    prompt_length = base_input_ids.shape[-1]

    image_sensitivity_scores = None
    image_sensitivity_info = {"available": False, "reason": "not_requested"}
    if use_token_disentangle and token_router in IMAGE_SENSITIVITY_ROUTERS:
        image_sensitivity_augmentation_mode = (
            "vppo_raw_image"
            if token_router in STRICT_VPPO_IMAGE_SENSITIVITY_ROUTERS
            else "processed_tensor"
        )
        image_sensitivity_scores, image_sensitivity_info = _compute_image_sensitivity_scores(
            model=model,
            inputs=inputs,
            full_input_ids=input_ids.to(device),
            prompt_length=prompt_length,
            start_index=start_index,
            update_length=update_length,
            drop_prob=image_sensitivity_drop_prob,
            augmentation_mode=image_sensitivity_augmentation_mode,
            processor=processor,
            input_text=input_text,
            media=media,
            patch_size=image_sensitivity_patch_size,
        )
        if image_sensitivity_info.get("available"):
            stats = image_sensitivity_info.get("score_stats") or {}
            print(
                colored(
                    f"-- Token router image sensitivity: available, "
                    f"mode={image_sensitivity_augmentation_mode}, "
                    f"drop_prob={float(image_sensitivity_drop_prob):.3f}, "
                    f"patch_size={int(image_sensitivity_patch_size)}, "
                    f"valid_tokens={image_sensitivity_info.get('valid_update_tokens')}, "
                    f"mean={stats.get('mean', float('nan')):.6f}, "
                    f"max={stats.get('max', float('nan')):.6f}",
                    "yellow",
                )
            )
        else:
            print(
                colored(
                    f"-- Token router image sensitivity unavailable: "
                    f"{image_sensitivity_info.get('reason', 'unknown')}; "
                    "falling back to reasoning tokens only.",
                    "yellow",
                )
            )

    reward_center = 0.5
    if evidence_reward is not None:
        reward_center = evidence_reward.reward_center
    elif visual_reward is not None and getattr(visual_reward, "reward_mode", "likelihood") == "contrastive":
        reward_center = 0.0

    current_pg_reward = orig_total_score - reward_center
    current_split_scores = _centered_evidence_scores(
        orig_reward_info,
        evidence_reward=evidence_reward,
        fallback_score=orig_total_score,
        reward_center=reward_center,
    )
    if use_visual_engagement_reward:
        current_split_scores["visual_score"] = 0.0
        current_split_scores["centered_total_score"] = float(current_split_scores["reasoning_score"])
        current_pg_reward = float(current_split_scores["centered_total_score"])
    current_reasoning_score = current_split_scores["reasoning_score"]
    current_visual_score = current_split_scores["visual_score"]

    for step in range(max_num_steps):
        optimizer.zero_grad()

        logits_from_hidden = model.lm_head(optimized_hidden_states)
        logit_device = logits_from_hidden.device
        probs = torch.softmax(logits_from_hidden, dim=-1) + 1e-8
        next_tokens = torch.argmax(probs, dim=-1).squeeze(-1)
        log_pi = torch.log(probs[torch.arange(update_length, device=logit_device), 0, next_tokens] + 1e-10)

        logit_probs = torch.softmax(logits_from_hidden.squeeze(1).float(), dim=-1)
        token_entropy = -(logit_probs * torch.log(logit_probs + 1e-10)).sum(dim=-1)
        reason_router_scores = None
        reason_router_signal_normalized = str(reason_router_signal or "entropy").strip().lower()
        if reason_router_signal_normalized == "neg_logprob":
            reason_router_scores = -log_pi.detach()
        elif reason_router_signal_normalized == "token_loss":
            target_start = prompt_length + start_index
            target_end = min(target_start + update_length, int(input_ids.shape[-1]))
            token_loss_scores = torch.zeros(update_length, device=logit_device, dtype=torch.float32)
            if target_end > target_start:
                target_ids = input_ids[:, target_start:target_end].to(logit_device).view(-1)
                valid_length = min(int(target_ids.numel()), int(update_length))
                if valid_length > 0:
                    token_loss_scores[:valid_length] = F.cross_entropy(
                        logits_from_hidden[:valid_length, 0, :].float(),
                        target_ids[:valid_length],
                        reduction="none",
                    ).detach()
            reason_router_scores = token_loss_scores
        elif reason_router_signal_normalized == "random":
            reason_router_scores = torch.rand(update_length, device=logit_device, dtype=torch.float32)
        elif reason_router_signal_normalized != "entropy":
            reason_router_signal_normalized = "entropy"
        router_info = None
        if use_token_disentangle:
            router_info = _compute_token_router_weights(
                optimized_hidden_states=optimized_hidden_states,
                token_entropy=token_entropy,
                visual_prototype=visual_prototype,
                image_sensitivity_scores=image_sensitivity_scores,
                reason_router_scores=reason_router_scores,
                reason_router_signal=reason_router_signal_normalized,
                token_router=token_router,
                visual_threshold=visual_router_threshold,
                entropy_threshold=entropy_router_threshold,
                router_temperature=router_temperature,
                visual_topk_ratio=visual_topk_ratio,
                reason_topk_ratio=reason_topk_ratio,
                visual_span_start_ratio=visual_span_start_ratio,
                visual_span_end_ratio=visual_span_end_ratio,
                reason_span_start_ratio=reason_span_start_ratio,
                reason_span_end_ratio=reason_span_end_ratio,
            )
            w_reason = router_info["w_reason"].to(log_pi.device)
            w_visual = router_info["w_visual"].to(log_pi.device)
            policy_loss = (
                -float(reason_policy_weight) * current_reasoning_score * (w_reason * log_pi).sum()
                -float(visual_policy_weight) * current_visual_score * (w_visual * log_pi).sum()
            )
        else:
            w_reason = None
            w_visual = None
            policy_loss = -current_pg_reward * log_pi.sum()

        logit_loss = torch.tensor(0.0, device=logit_device)
        if dual_space:
            if use_token_disentangle and w_reason is not None:
                logit_loss = logit_lr * _safe_weighted_mean(token_entropy, w_reason)
            else:
                logit_loss = logit_lr * token_entropy.mean()

        anchor_loss = torch.tensor(0.0, device=logit_device)
        if anchor_weight > 0:
            anchor_loss = anchor_weight * F.mse_loss(optimized_hidden_states, original_hidden_slice)
        if use_token_disentangle and visual_anchor_extra > 0 and w_visual is not None:
            token_mse = (optimized_hidden_states - original_hidden_slice).pow(2).mean(dim=-1).view(-1)
            anchor_loss = anchor_loss + float(visual_anchor_extra) * _safe_weighted_mean(token_mse, w_visual)

        loss = policy_loss + logit_loss + anchor_loss
        loss_record = {
            "step": int(step),
            "pg_reward_before_step": float(current_pg_reward),
            "reasoning_score_before_step": float(current_reasoning_score),
            "visual_score_before_step": float(current_visual_score),
            "reward_center": float(reward_center),
            "log_pi_sum": float(log_pi.sum().item()),
            "policy_loss": float(policy_loss.item()),
            "logit_loss": float(logit_loss.item()),
            "anchor_loss": float(anchor_loss.item()),
            "total_loss": float(loss.item()),
            "use_token_disentangle": bool(use_token_disentangle),
        }
        if router_info is not None:
            loss_record["token_router"] = router_info["stats"]
            loss_record["weighted_log_pi_reason"] = float((w_reason * log_pi).sum().item())
            loss_record["weighted_log_pi_visual"] = float((w_visual * log_pi).sum().item())
        loss_history_detailed.append(loss_record)
        if debug_loss_terms:
            router_msg = ""
            if router_info is not None:
                stats = router_info["stats"]
                router_msg = (
                    f", reason_score={current_reasoning_score:.6f}, "
                    f"visual_score={current_visual_score:.6f}, "
                    f"w_reason_mean={stats['w_reason']['mean']:.4f}, "
                    f"w_visual_mean={stats['w_visual']['mean']:.4f}"
                )
            print(
                colored(
                    f"-- Loss step {step}: "
                    f"pg_reward={current_pg_reward:.6f}, "
                    f"reward_center={reward_center:.6f}, "
                    f"log_pi_sum={loss_record['log_pi_sum']:.6f}, "
                    f"policy={policy_loss.item():.6f}, "
                    f"logit={logit_loss.item():.6f}, "
                    f"anchor={anchor_loss.item():.6f}, "
                    f"total={loss.item():.6f}"
                    f"{router_msg}",
                    "magenta",
                )
            )
        loss.backward(retain_graph=True)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_([optimized_hidden_states], grad_clip)
        optimizer.step()

        if debug_visual_focus and visual_focus_diagnostics is not None:
            optimized_focus = _visual_focus_block(optimized_hidden_states, visual_prototype)
            optimized_mean = _mean_from_focus_block(optimized_focus)
            if optimized_mean is not None and original_update_focus_mean is not None:
                delta_mean = optimized_mean - original_update_focus_mean
            else:
                delta_mean = None
            visual_focus_diagnostics["steps"].append(
                {
                    "step": int(step),
                    "optimized_update_slice": optimized_focus,
                    "delta_score_mean_from_original_update": delta_mean,
                }
            )
            if optimized_mean is not None:
                print(
                    colored(
                        f"-- Visual focus step {step}: update_mean={optimized_mean:.4f}, "
                        f"delta_from_original={delta_mean if delta_mean is not None else float('nan'):.4f}",
                        "yellow",
                    )
                )

        step_candidates = []
        step_candidate_traces = []
        with torch.no_grad():
            greedy_logits = model.lm_head(optimized_hidden_states).squeeze(1)
            greedy_tokens = torch.argmax(greedy_logits, dim=-1)

            greedy_trace = _generate_from_hidden_prefix(
                model,
                tokenizer,
                prompt_length,
                base_input_ids_trimmed,
                greedy_tokens,
                inputs,
                max_new_tokens,
                return_trace=use_visual_engagement_reward,
            )
            greedy_answer = greedy_trace["text"] if use_visual_engagement_reward else greedy_trace
            step_candidates.append(greedy_answer)
            if use_visual_engagement_reward:
                step_candidate_traces.append(greedy_trace)

            for nc in range(noise_candidates):
                scale = noise_std * (1.0 + nc * 0.5)
                noise = torch.randn_like(optimized_hidden_states) * scale
                noisy_hidden = optimized_hidden_states + noise
                noisy_logits = model.lm_head(noisy_hidden).squeeze(1)

                topk = min(8, noisy_logits.shape[-1])
                topk_vals, topk_idx = torch.topk(noisy_logits, k=topk, dim=-1)
                topk_probs = torch.softmax(topk_vals / max(0.6 + nc * 0.2, 0.3), dim=-1)
                sampled_local = torch.multinomial(topk_probs, num_samples=1).squeeze(-1)
                noisy_tokens = topk_idx.gather(1, sampled_local.unsqueeze(-1)).squeeze(-1)

                noisy_trace = _generate_from_hidden_prefix(
                    model,
                    tokenizer,
                    prompt_length,
                    base_input_ids_trimmed,
                    noisy_tokens,
                    inputs,
                    max_new_tokens,
                    return_trace=use_visual_engagement_reward,
                )
                noisy_answer = noisy_trace["text"] if use_visual_engagement_reward else noisy_trace
                step_candidates.append(noisy_answer)
                if use_visual_engagement_reward:
                    step_candidate_traces.append(noisy_trace)

        step_reward_infos = None
        if evidence_reward is not None:
            step_reward_infos = evidence_reward.score_candidates(
                question=question,
                answer_texts=step_candidates,
                media=media,
                problem=problem,
                clues=visual_clues,
                crop_evidence=crop_evidence,
                reference_answer=original_answer,
            )

        step_visual_engagement_infos = []
        if use_visual_engagement_reward:
            for cand_idx, trace in enumerate(step_candidate_traces):
                scored_ids = _cap_generated_ids_for_visual_engagement(
                    trace.get("generated_ids"),
                    max_score_tokens=visual_engagement_max_score_tokens,
                )
                engagement_info = visual_engagement_score(
                    model=visual_engagement_model,
                    tokenizer=visual_engagement_tokenizer,
                    inputs=visual_engagement_inputs,
                    generated_ids=scored_ids,
                    layer_index=visual_engagement_layer_index,
                )
                reward_info = visual_engagement_reward(
                    current_engagement=engagement_info["engagement"],
                    baseline_engagement=visual_engagement_baseline,
                    tau=visual_engagement_tau,
                )
                combined_info = {
                    "candidate_index": int(cand_idx),
                    "engagement": float(engagement_info["engagement"]),
                    "delta": float(reward_info["delta"]),
                    "reward": float(reward_info["reward"]),
                    "tau": float(reward_info["tau"]),
                    "num_generated_tokens": int(engagement_info.get("num_generated_tokens", 0)),
                    "num_image_tokens": int(engagement_info.get("num_image_tokens", 0)),
                    "reason": engagement_info.get("reason"),
                }
                step_visual_engagement_infos.append(combined_info)
                if step_reward_infos is not None and cand_idx < len(step_reward_infos):
                    if evidence_reward is not None:
                        ecl_visual_score = (
                            evidence_reward.clue_weight * float(step_reward_infos[cand_idx].get("clue_margin_score", 0.0))
                            + evidence_reward.crop_weight * float(step_reward_infos[cand_idx].get("crop_reward", 0.0))
                        )
                    else:
                        ecl_visual_score = 0.0
                    step_reward_infos[cand_idx]["ecl_visual_score"] = float(ecl_visual_score)
                    step_reward_infos[cand_idx]["visual_engagement"] = combined_info["engagement"]
                    step_reward_infos[cand_idx]["visual_engagement_delta"] = combined_info["delta"]
                    step_reward_infos[cand_idx]["visual_engagement_reward"] = combined_info["reward"]
                    step_reward_infos[cand_idx]["visual_engagement_num_tokens"] = combined_info["num_generated_tokens"]
                    step_reward_infos[cand_idx]["visual_engagement_num_image_tokens"] = combined_info["num_image_tokens"]

            visual_engagement_history.append(
                {
                    "step": int(step),
                    "baseline_engagement": float(visual_engagement_baseline),
                    "candidates": step_visual_engagement_infos,
                }
            )

        conditioned_info_by_candidate_index = {}
        if debug_candidate_conditioned_clues and step_reward_infos is not None:
            ranked_indices = sorted(
                range(len(step_candidates)),
                key=lambda item: step_reward_infos[item].get("total_score", float("-inf")),
                reverse=True,
            )
            if candidate_conditioned_clue_top_k > 0:
                selected_indices = ranked_indices[:candidate_conditioned_clue_top_k]
            else:
                selected_indices = ranked_indices

            selected_answer_keys = []
            selected_clues = []
            selected_records = []
            for rank, cand_idx in enumerate(selected_indices):
                cand_text = step_candidates[cand_idx]
                answer_key = _conditioned_clue_answer_key(cand_text, evidence_reward=evidence_reward, problem=problem)
                conditioned_clues = _get_candidate_conditioned_clues(
                    cache=candidate_conditioned_clue_cache,
                    model=model,
                    tokenizer=tokenizer,
                    processor=candidate_conditioned_clue_processor,
                    device=device,
                    problem=problem,
                    media=media,
                    answer_key=answer_key,
                    max_new_tokens=candidate_conditioned_clue_max_new_tokens,
                    num_clues=max(1, len(visual_clues or []) or 3),
                )
                selected_answer_keys.append(answer_key)
                selected_clues.append(conditioned_clues)
                selected_records.append(
                    {
                        "candidate_index": int(cand_idx),
                        "rank_by_total_score": int(rank),
                        "answer_key": answer_key,
                        "short_answer": None if _extract_short_answer(cand_text, problem) is None else str(_extract_short_answer(cand_text, problem)),
                        "text_preview": _preview_text(cand_text, limit=120),
                        "base_total_score": float(step_reward_infos[cand_idx].get("total_score", 0.0)),
                        "base_answer_reward": float(step_reward_infos[cand_idx].get("answer_reward", 0.0)),
                        "base_clue_margin_score": float(step_reward_infos[cand_idx].get("clue_margin_score", 0.0)),
                        "base_margin_score": float(step_reward_infos[cand_idx].get("margin_score", 0.0)),
                        "conditioned_clues": list(conditioned_clues),
                    }
                )

            conditioned_infos = evidence_reward.score_candidate_conditioned_clues(
                answer_texts=selected_answer_keys,
                candidate_clues=selected_clues,
                media=media,
                problem=problem,
                reference_answer=reference_conditioned_answer_key,
                reference_clues=reference_conditioned_clues,
            )
            step_conditioned_records = []
            for record, conditioned_info in zip(selected_records, conditioned_infos):
                enriched = dict(record)
                enriched.update(
                    {
                        "conditioned_clue_score": float(conditioned_info.get("clue_score", 0.0)),
                        "conditioned_clue_reward": float(conditioned_info.get("clue_reward", 0.0)),
                        "conditioned_clue_margin_delta": float(conditioned_info.get("conditioned_clue_margin_delta", 0.0)),
                        "conditioned_clue_margin_score": float(conditioned_info.get("conditioned_clue_margin_score", 0.0)),
                        "conditioned_competitor_clue_baseline": float(conditioned_info.get("conditioned_competitor_clue_baseline", 0.0)),
                        "conditioned_clue_visual_logprob": conditioned_info.get("clue_visual_logprob"),
                        "conditioned_clue_prior_logprob": conditioned_info.get("clue_prior_logprob"),
                        "conditioned_clue_text": conditioned_info.get("clue_text"),
                    }
                )
                conditioned_info_by_candidate_index[record["candidate_index"]] = enriched
                step_conditioned_records.append(enriched)

            if step_conditioned_records:
                candidate_conditioned_clue_history.append(
                    {
                        "step": int(step),
                        "top_k": int(candidate_conditioned_clue_top_k),
                        "reference": {
                            "answer_key": reference_conditioned_answer_key,
                            "conditioned_clues": list(reference_conditioned_clues),
                            "conditioned_clue_score": None if reference_conditioned_info is None else float(reference_conditioned_info.get("clue_score", 0.0)),
                            "conditioned_clue_reward": None if reference_conditioned_info is None else float(reference_conditioned_info.get("clue_reward", 0.0)),
                            "conditioned_clue_text": None if reference_conditioned_info is None else reference_conditioned_info.get("clue_text"),
                        },
                        "candidates": step_conditioned_records,
                    }
                )
                best_conditioned = max(
                    step_conditioned_records,
                    key=lambda item: item.get("conditioned_clue_margin_score", float("-inf")),
                )
                print(
                    colored(
                        f"-- Step {step} candidate-conditioned clues: "
                        f"best_idx={best_conditioned['candidate_index']}, "
                        f"score={best_conditioned['conditioned_clue_score']:.4f}, "
                        f"margin={best_conditioned['conditioned_clue_margin_score']:.4f}, "
                        f"num_clues={len(best_conditioned.get('conditioned_clues') or [])}",
                        "yellow",
                    )
                )

        step_best_score = float("-inf")
        step_best_info = None
        step_best_idx = None
        step_candidate_records = []
        for idx, cand_text in enumerate(step_candidates):
            short = _extract_short_answer(cand_text, problem)
            if step_reward_infos is not None:
                reward_info = step_reward_infos[idx]
                score = reward_info["total_score"]
            elif visual_reward is not None and media is not None:
                reward_info = None
                score = visual_reward.get_reward(question, cand_text, media=media, problem=problem)
            else:
                reward_info = None
                score = 0.0
            all_candidates.append((cand_text, short, score))
            if reward_info is not None and debug_candidate_rewards:
                candidate_record = {
                    "candidate_index": int(idx),
                    "short_answer": None if short is None else str(short),
                    "text_preview": _preview_text(cand_text, limit=120),
                    "total_score": float(reward_info["total_score"]),
                    "answer_score": float(reward_info.get("answer_score", 0.0)),
                    "answer_reward": float(reward_info.get("answer_reward", 0.0)),
                    "answer_margin_delta": float(reward_info.get("answer_margin_delta", 0.0)),
                    "clue_score": float(reward_info.get("clue_score", 0.0)),
                    "clue_reward": float(reward_info.get("clue_reward", 0.0)),
                    "clue_margin_delta": float(reward_info.get("clue_margin_delta", 0.0)),
                    "clue_margin_score": float(reward_info.get("clue_margin_score", 0.0)),
                    "crop_evidence_score": float(reward_info.get("crop_evidence_score", 0.0)),
                    "crop_reward": float(reward_info.get("crop_reward", 0.0)),
                    "support_gap": float(reward_info.get("support_gap", 0.0)),
                    "mask_drop": float(reward_info.get("mask_drop", 0.0)),
                    "margin_score": float(reward_info.get("margin_score", 0.0)),
                }
                if use_visual_engagement_reward and idx < len(step_visual_engagement_infos):
                    candidate_record["visual_engagement"] = step_visual_engagement_infos[idx]
                    candidate_record["ecl_visual_score"] = float(reward_info.get("ecl_visual_score", 0.0))
                if idx in conditioned_info_by_candidate_index:
                    candidate_record["candidate_conditioned_clues"] = conditioned_info_by_candidate_index[idx]
                step_candidate_records.append(candidate_record)
                print(
                    colored(
                        f"-- Step {step} cand {idx}: short={short}, "
                        + _format_evidence_reward_breakdown(reward_info, evidence_reward=evidence_reward),
                        "cyan",
                    )
                )
            if score > step_best_score:
                step_best_score = score
                step_best_info = reward_info
                step_best_idx = int(idx)

        current_pg_reward = step_best_score - reward_center
        current_split_scores = _centered_evidence_scores(
            step_best_info,
            evidence_reward=evidence_reward,
            fallback_score=step_best_score,
            reward_center=reward_center,
        )
        selected_visual_engagement_info = None
        if use_visual_engagement_reward and step_best_idx is not None and step_best_idx < len(step_visual_engagement_infos):
            selected_visual_engagement_info = step_visual_engagement_infos[step_best_idx]
            current_split_scores["visual_score"] = float(selected_visual_engagement_info["reward"])
            current_split_scores["centered_total_score"] = float(
                current_split_scores["reasoning_score"] + current_split_scores["visual_score"]
            )
            current_split_scores["clue_term"] = 0.0
            current_split_scores["crop_term"] = 0.0
        answer_likelihood_delta_info = None
        if (
            use_answer_likelihood_delta_visual_reward
            and step_best_info is not None
            and orig_reward_info is not None
        ):
            baseline_answer_score = float(orig_reward_info.get("answer_score", 0.0))
            current_answer_score = float(step_best_info.get("answer_score", 0.0))
            answer_score_delta = current_answer_score - baseline_answer_score
            answer_delta_tau = 1.0
            if visual_reward is not None:
                answer_delta_tau = max(
                    float(getattr(visual_reward, "contrastive_temperature", 1.0)),
                    1e-6,
                )
            answer_delta_reward = math.tanh(answer_score_delta / answer_delta_tau)
            answer_likelihood_delta_info = {
                "baseline_answer_score": float(baseline_answer_score),
                "current_answer_score": float(current_answer_score),
                "delta": float(answer_score_delta),
                "tau": float(answer_delta_tau),
                "reward": float(answer_delta_reward),
            }
            current_split_scores["visual_score"] = float(answer_delta_reward)
            current_split_scores["centered_total_score"] = float(
                current_split_scores["reasoning_score"] + current_split_scores["visual_score"]
            )
            current_split_scores["clue_term"] = 0.0
            current_split_scores["crop_term"] = 0.0
            step_best_info["answer_likelihood_visual_delta"] = float(answer_score_delta)
            step_best_info["answer_likelihood_visual_reward"] = float(answer_delta_reward)
        if use_visual_engagement_reward or use_answer_likelihood_delta_visual_reward:
            current_pg_reward = float(current_split_scores["centered_total_score"])
        current_reasoning_score = current_split_scores["reasoning_score"]
        current_visual_score = current_split_scores["visual_score"]
        reward_history.append(step_best_score)
        reward_history_detailed.append(
            {
                "step": step,
                "best_total_score": float(step_best_score),
                "best_centered_total_score": float(current_split_scores["centered_total_score"]),
                "best_reasoning_score": float(current_split_scores["reasoning_score"]),
                "best_visual_score": float(current_split_scores["visual_score"]),
                "best_answer_term": float(current_split_scores["answer_term"]),
                "best_margin_term": float(current_split_scores["margin_term"]),
                "best_clue_term": float(current_split_scores["clue_term"]),
                "best_crop_term": float(current_split_scores["crop_term"]),
                "best_answer_score": None if step_best_info is None else float(step_best_info["answer_score"]),
                "best_answer_reward": None if step_best_info is None else float(step_best_info["answer_reward"]),
                "best_clue_score": None if step_best_info is None else float(step_best_info["clue_score"]),
                "best_clue_reward": None if step_best_info is None else float(step_best_info["clue_reward"]),
                "best_answer_margin_delta": None if step_best_info is None else float(step_best_info.get("answer_margin_delta", 0.0)),
                "best_clue_margin_delta": None if step_best_info is None else float(step_best_info.get("clue_margin_delta", 0.0)),
                "best_clue_margin_score": None if step_best_info is None else float(step_best_info.get("clue_margin_score", 0.0)),
                "best_crop_evidence_score": None if step_best_info is None else float(step_best_info["crop_evidence_score"]),
                "best_crop_reward": None if step_best_info is None else float(step_best_info["crop_reward"]),
                "best_support_gap": None if step_best_info is None else float(step_best_info.get("support_gap", 0.0)),
                "best_mask_drop": None if step_best_info is None else float(step_best_info.get("mask_drop", 0.0)),
                "best_margin_score": None if step_best_info is None else float(step_best_info["margin_score"]),
                "competitor_best_reward": None if step_best_info is None else float(step_best_info["competitor_best_reward"]),
                "best_visual_engagement": None if selected_visual_engagement_info is None else float(selected_visual_engagement_info["engagement"]),
                "best_visual_engagement_delta": None if selected_visual_engagement_info is None else float(selected_visual_engagement_info["delta"]),
                "best_visual_engagement_reward": None if selected_visual_engagement_info is None else float(selected_visual_engagement_info["reward"]),
                "best_ecl_visual_score": None if step_best_info is None else float(step_best_info.get("ecl_visual_score", current_split_scores["visual_score"])),
                "best_answer_likelihood_visual_delta": None if answer_likelihood_delta_info is None else float(answer_likelihood_delta_info["delta"]),
                "best_answer_likelihood_visual_reward": None if answer_likelihood_delta_info is None else float(answer_likelihood_delta_info["reward"]),
            }
        )
        if step_candidate_records:
            candidate_reward_history.append(
                {
                    "step": int(step),
                    "candidates": step_candidate_records,
                }
            )

        del greedy_logits, greedy_tokens
        torch.cuda.empty_cache()

        step_shorts = [_extract_short_answer(c, problem) for c in step_candidates]
        unique = set(str(s).lower() if s else "none" for s in step_shorts)
        if step_best_info is not None:
            visual_msg = ""
            if selected_visual_engagement_info is not None:
                visual_msg = (
                    f"AVER={selected_visual_engagement_info['reward']:.4f}, "
                    f"AVER_delta={selected_visual_engagement_info['delta']:.4f}, "
                )
            elif answer_likelihood_delta_info is not None:
                visual_msg = (
                    f"answer_like_visual={answer_likelihood_delta_info['reward']:.4f}, "
                    f"answer_like_delta={answer_likelihood_delta_info['delta']:.4f}, "
                )
            print(
                colored(
                    f"-- Step {step}: best_total={step_best_score:.4f}, "
                    f"best_answer_reward={step_best_info['answer_reward']:.4f}, "
                    f"best_answer_margin_delta={step_best_info.get('answer_margin_delta', 0.0):.4f}, "
                    f"best_clue_reward={step_best_info['clue_reward']:.4f}, "
                    f"best_clue_margin_delta={step_best_info.get('clue_margin_delta', 0.0):.4f}, "
                    f"best_clue_margin={step_best_info.get('clue_margin_score', 0.0):.4f}, "
                    f"best_crop_reward={step_best_info['crop_reward']:.4f}, "
                    f"best_support_gap={step_best_info.get('support_gap', 0.0):.4f}, "
                    f"best_mask_drop={step_best_info.get('mask_drop', 0.0):.4f}, "
                    f"best_margin={step_best_info['margin_score']:.4f}, "
                    f"{visual_msg}"
                    f"pg_reward={current_pg_reward:.4f}, "
                    f"unique_answers={len(unique)}/{len(step_candidates)} {unique}",
                    "blue",
                )
            )
            if debug_reward_terms:
                print(
                    colored(
                        f"-- Step {step} reward breakdown: "
                        + _format_evidence_reward_breakdown(step_best_info, evidence_reward=evidence_reward),
                        "cyan",
                    )
                )
        else:
            print(
                colored(
                    f"-- Step {step}: best_visual={step_best_score:.4f}, "
                    f"pg_reward={current_pg_reward:.4f}, "
                    f"unique_answers={len(unique)}/{len(step_candidates)} {unique}",
                    "blue",
                )
            )

    valid = [(t, s, c) for t, s, c in all_candidates if s is not None]
    if not valid:
        valid = all_candidates

    final_answer, final_short = _select_by_consistency_and_score(
        valid,
        mode=candidate_aggregation,
        score_temperature=aggregation_temperature,
    )

    shorts = [s for _, s, _ in valid if s is not None]
    if shorts:
        counter = Counter(str(s).strip().lower() for s in shorts)
        top3 = counter.most_common(3)
        print(colored(f"-- Self-consistency: {top3}, selected: {final_short}", "green"))

    if final_answer is None:
        final_answer = original_answer

    optimized_length = len(tokenizer.encode(final_answer))
    print(f"-- Final: {final_answer[:120]}...")
    if reward_diagnostics is not None:
        reward_formula = {
            "reward_center": float(reward_center),
            "answer_weight": None if evidence_reward is None else float(evidence_reward.answer_weight),
            "clue_weight": None if evidence_reward is None else float(evidence_reward.clue_weight),
            "crop_weight": None if evidence_reward is None else float(evidence_reward.crop_weight),
            "margin_weight": None if evidence_reward is None else float(evidence_reward.margin_weight),
            "total_score_formula": (
                "total_score = answer_weight*answer_reward + clue_weight*clue_margin_score + crop_weight*crop_reward + margin_weight*margin_score"
                if evidence_reward is not None
                else "total_score = visual_reward"
            ),
            "loss_formula": "loss = policy_loss + logit_loss + anchor_loss",
            "policy_loss_formula": (
                "policy_loss = -reasoning_score * sum(w_reason * log_pi) - visual_score * sum(w_visual * log_pi)"
                if use_token_disentangle
                else "policy_loss = -pg_reward * sum(log_pi)"
            ),
            "reasoning_score_formula": "reasoning_score = answer_weight*(answer_reward-answer_center) + margin_weight*margin_score",
            "visual_score_formula": (
                "visual_score = tanh((E(candidate_trace)-E(original_trace))/visual_engagement_tau), "
                "E=mean generated-token attention mass to image tokens; applied only to visual-routed tokens"
                if use_visual_engagement_reward
                else (
                    "visual_score = tanh((answer_score(candidate)-answer_score(original))/temperature), "
                    "answer_score is image-conditioned short-answer likelihood"
                    if use_answer_likelihood_delta_visual_reward
                    else "visual_score = clue_weight*clue_margin_score + crop_weight*crop_reward"
                )
            ),
            "visual_branch_reward_mode": str(visual_branch_reward_mode),
            "visual_engagement_tau": float(visual_engagement_tau),
            "visual_engagement_layer_index": str(visual_engagement_layer_index),
            "visual_engagement_max_score_tokens": int(visual_engagement_max_score_tokens),
            "token_router_formula": (
                "soft: w_visual = sigmoid((visual_focus - visual_threshold) / temperature); "
                "w_reason = sigmoid((entropy_z - entropy_threshold) / temperature) * (1 - w_visual). "
                "hard_topk: visual=top visual_focus tokens, reason=top entropy tokens outside visual tokens. "
                "vppo_overlap_topk: visual=image-sensitive tokens, reason=entropy tokens with overlap allowed. "
                "vppo_swapped_topk: visual=entropy tokens, reason=image-sensitive tokens outside visual tokens. "
                "random_topk/uniform_topk: replace token scores with random or evenly spaced token selection. "
                "span_constrained: route only inside configured relative update spans."
            ),
            "logit_loss_formula": (
                "logit_loss = logit_lr * weighted_mean(token_entropy, w_reason) when token_disentangle else logit_lr * mean(token_entropy)"
            ),
            "anchor_loss_formula": "anchor_loss = anchor_weight * mse(optimized_hidden_states, original_hidden_slice) when anchor_weight > 0 else 0",
            "visual_focus_formula": "visual_focus_score = 0.5 * (cosine(hidden_state, mean(image_token_hidden_states)) + 1); diagnostics only, not included in total_score",
            "candidate_conditioned_clue_formula": "diagnostics only: each selected candidate first generates its own visual clues, then the image-conditioned clue support score is compared against selected siblings and the original-answer reference",
            "use_token_disentangle": bool(use_token_disentangle),
            "token_router": token_router,
            "visual_router_threshold": float(visual_router_threshold),
            "entropy_router_threshold": float(entropy_router_threshold),
            "router_temperature": float(router_temperature),
            "visual_topk_ratio": float(visual_topk_ratio),
            "reason_topk_ratio": float(reason_topk_ratio),
            "visual_span_start_ratio": float(visual_span_start_ratio),
            "visual_span_end_ratio": float(visual_span_end_ratio),
            "reason_span_start_ratio": float(reason_span_start_ratio),
            "reason_span_end_ratio": float(reason_span_end_ratio),
            "visual_policy_weight": float(visual_policy_weight),
            "reason_policy_weight": float(reason_policy_weight),
            "visual_anchor_extra": float(visual_anchor_extra),
        }
        reward_diagnostics.clear()
        reward_diagnostics["reward_history"] = [float(item) for item in reward_history]
        reward_diagnostics["reward_history_detailed"] = reward_history_detailed
        reward_diagnostics["loss_history_detailed"] = loss_history_detailed
        reward_diagnostics["reward_formula"] = reward_formula
        reward_diagnostics["reward_center"] = float(reward_center)
        reward_diagnostics["initial_reward_info"] = orig_reward_info
        reward_diagnostics["visual_clues"] = list(visual_clues or [])
        reward_diagnostics["crop_evidence"] = None if crop_evidence is None else crop_evidence.get("summary")
        if use_visual_engagement_reward:
            reward_diagnostics["visual_engagement"] = {
                "baseline": visual_engagement_baseline_info,
                "history": visual_engagement_history,
            }
        if image_sensitivity_info is not None and image_sensitivity_info.get("reason") != "not_requested":
            reward_diagnostics["image_sensitivity_router"] = image_sensitivity_info
        if visual_focus_diagnostics is not None:
            reward_diagnostics["visual_focus_diagnostics"] = visual_focus_diagnostics
        if candidate_conditioned_clue_history:
            reward_diagnostics["candidate_conditioned_clue_history"] = candidate_conditioned_clue_history
        if candidate_reward_history:
            reward_diagnostics["candidate_reward_history"] = candidate_reward_history
        reward_diagnostics["final_selection"] = {
            "final_answer": final_answer,
            "final_short": final_short,
            "candidate_aggregation": candidate_aggregation,
        }
    return final_answer, reward_history, original_length, optimized_length, update_length
