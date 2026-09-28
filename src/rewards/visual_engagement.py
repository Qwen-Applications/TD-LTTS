"""Answer-agnostic visual engagement reward utilities."""

import math

import torch

from methods.latentgrad.generation_utils import (
    GENERATION_MEDIA_KEYS,
    build_generation_kwargs,
    get_stop_token_ids,
    reset_multimodal_generation_state,
    trim_after_stop_words,
)


STOP_WORDS = ["</s>", "<|im_end|>", "<|endoftext|>"]


def candidate_visual_token_ids(model, tokenizer):
    """Return token ids used as image placeholders by supported VLMs."""
    token_ids = set()
    config = getattr(model, "config", None)
    for attr in ("image_token_id", "video_token_id"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value >= 0:
            token_ids.add(value)

    for token in ("<|image_pad|>", "<image>", "<IMG_CONTEXT>"):
        try:
            value = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            value = None
        if isinstance(value, int) and value >= 0 and value != getattr(tokenizer, "unk_token_id", None):
            token_ids.add(value)
    return sorted(token_ids)


def image_token_mask(input_ids, model, tokenizer):
    """Boolean mask over input positions that correspond to image tokens."""
    token_ids = candidate_visual_token_ids(model, tokenizer)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in token_ids:
        mask |= input_ids == token_id
    return mask


def _as_stop_id_set(tokenizer):
    stop_token_ids = get_stop_token_ids(tokenizer)
    if stop_token_ids is None:
        return set()
    if isinstance(stop_token_ids, int):
        return {stop_token_ids}
    return set(int(item) for item in stop_token_ids)


def strip_stop_tokens(token_ids, tokenizer):
    """Drop trailing generation at the first stop token/string."""
    stop_ids = _as_stop_id_set(tokenizer)
    kept = []
    for token_id in token_ids:
        token_id = int(token_id)
        piece = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if token_id in stop_ids or piece in STOP_WORDS:
            break
        kept.append(token_id)
    return kept


def generate_trace(
    model,
    tokenizer,
    inputs,
    max_new_tokens,
    do_sample=False,
    temperature=0.7,
    top_p=0.95,
):
    """Generate one trace and return non-stop token ids plus decoded text."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    reset_multimodal_generation_state(model)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": bool(do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": get_stop_token_ids(tokenizer),
        "return_dict_in_generate": True,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": float(temperature),
                "top_p": float(top_p),
            }
        )

    with torch.no_grad():
        output = model.generate(**inputs, **generation_kwargs)

    prompt_length = int(inputs["input_ids"].shape[-1])
    generated_ids_all = output.sequences[0, prompt_length:].detach().cpu().tolist()
    generated_ids = strip_stop_tokens(generated_ids_all, tokenizer)
    raw_text = tokenizer.decode(
        generated_ids_all,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    text = trim_after_stop_words(text, STOP_WORDS)
    return {
        "text": text,
        "raw_text": trim_after_stop_words(raw_text, STOP_WORDS),
        "generated_ids": generated_ids,
        "generated_ids_all": generated_ids_all,
        "prompt_length": prompt_length,
    }


def _build_forward_inputs(inputs, full_input_ids):
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
    return forward_inputs


def visual_engagement_score(
    model,
    tokenizer,
    inputs,
    generated_ids,
    layer_index=-1,
):
    """Compute AVER-v1 E(y): mean generated-token attention mass to image tokens.

    The score is answer-agnostic: it only uses the model's last-layer attention
    from generated-token rows to image-token columns.
    """
    prompt_input_ids = inputs["input_ids"]
    prompt_length = int(prompt_input_ids.shape[-1])
    if not generated_ids:
        return {
            "engagement": 0.0,
            "num_generated_tokens": 0,
            "num_image_tokens": int(image_token_mask(prompt_input_ids, model, tokenizer).sum().item()),
            "per_token": [],
            "layer_index": str(layer_index),
            "reason": "empty_generation",
        }

    generated_tensor = torch.tensor(
        [generated_ids],
        dtype=prompt_input_ids.dtype,
        device=prompt_input_ids.device,
    )
    full_input_ids = torch.cat([prompt_input_ids, generated_tensor], dim=-1)
    img_mask = image_token_mask(full_input_ids, model, tokenizer)
    image_positions = torch.nonzero(img_mask[0], as_tuple=False).view(-1)
    if image_positions.numel() == 0:
        return {
            "engagement": 0.0,
            "num_generated_tokens": int(len(generated_ids)),
            "num_image_tokens": 0,
            "per_token": [],
            "layer_index": str(layer_index),
            "reason": "no_image_tokens",
        }

    forward_inputs = _build_forward_inputs(inputs, full_input_ids)
    reset_multimodal_generation_state(model)
    with torch.no_grad():
        outputs = model(
            **forward_inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    attentions = getattr(outputs, "attentions", None)
    if not attentions:
        raise RuntimeError(
            "Model did not return attentions. Load with --attn_implementation eager "
            "or use a model backend that supports output_attentions=True."
        )
    layer_spec = str(layer_index).strip().lower()
    average_layers = layer_spec in {"all", "avg", "average", "mean"}
    if average_layers:
        selected_layers = list(range(len(attentions)))
        selected_layer_label = "all"
    else:
        selected_layers = [int(layer_index)]
        selected_layer_label = str(int(layer_index))

    first_attn = attentions[selected_layers[0]]
    if first_attn is None:
        raise RuntimeError(f"Attention tensor at layer {selected_layer_label} is None.")
    if first_attn.ndim != 4:
        raise RuntimeError(f"Expected attention shape [batch, heads, query, key], got {tuple(first_attn.shape)}.")

    query_len = int(first_attn.shape[-2])
    key_len = int(first_attn.shape[-1])
    answer_start = min(prompt_length, query_len)
    answer_end = min(prompt_length + len(generated_ids), query_len)
    image_positions = image_positions[image_positions < key_len]
    if answer_end <= answer_start or image_positions.numel() == 0:
        return {
            "engagement": 0.0,
            "num_generated_tokens": int(len(generated_ids)),
            "num_image_tokens": int(image_positions.numel()),
            "per_token": [],
            "layer_index": selected_layer_label,
            "reason": "empty_attention_slice",
        }

    per_layer = []
    for selected_layer in selected_layers:
        attn = attentions[selected_layer]
        if attn is None:
            raise RuntimeError(f"Attention tensor at layer {selected_layer} is None.")
        if attn.ndim != 4:
            raise RuntimeError(f"Expected attention shape [batch, heads, query, key], got {tuple(attn.shape)}.")
        token_to_image = attn[0, :, answer_start:answer_end, :][:, :, image_positions]
        per_layer.append(token_to_image.float().sum(dim=-1).mean(dim=0))
    per_token = torch.stack(per_layer, dim=0).mean(dim=0)
    score = float(per_token.mean().item())
    return {
        "engagement": score,
        "num_generated_tokens": int(answer_end - answer_start),
        "num_image_tokens": int(image_positions.numel()),
        "per_token": [float(item) for item in per_token.detach().cpu().tolist()],
        "per_token_min": float(per_token.min().item()),
        "per_token_max": float(per_token.max().item()),
        "per_token_std": float(per_token.std(unbiased=False).item()) if per_token.numel() > 1 else 0.0,
        "layer_index": selected_layer_label,
        "num_layers_scored": int(len(selected_layers)),
        "reason": "ok",
    }


def visual_engagement_reward(current_engagement, baseline_engagement, tau=0.05):
    """Compute tanh-normalized delta reward against a per-example baseline."""
    tau = max(float(tau), 1e-8)
    delta = float(current_engagement) - float(baseline_engagement)
    reward = math.tanh(delta / tau)
    return {
        "delta": delta,
        "tau": tau,
        "reward": float(reward),
    }
