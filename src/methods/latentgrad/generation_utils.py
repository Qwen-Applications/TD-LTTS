import torch


GENERATION_MEDIA_KEYS = [
    "pixel_values",
    "pixel_values_videos",
    "image_flags",
    "image_sizes",
    "image_grid_thw",
    "video_grid_thw",
    "second_per_grid_ts",
]


def move_model_inputs(model_inputs, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in model_inputs.items()
    }


def get_stop_token_ids(tokenizer):
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    for token in ["<|im_end|>", "</s>", "<|endoftext|>"]:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            token_id = None
        if token_id is not None and token_id >= 0 and token_id not in stop_token_ids:
            stop_token_ids.append(token_id)

    if not stop_token_ids:
        return None
    if len(stop_token_ids) == 1:
        return stop_token_ids[0]
    return stop_token_ids


def trim_after_stop_words(text, stop_words):
    if text is None:
        return None

    cutoff = None
    for stop_word in stop_words:
        if not stop_word:
            continue
        idx = text.find(stop_word)
        if idx != -1:
            cutoff = idx if cutoff is None else min(cutoff, idx)

    if cutoff is None:
        return text
    return text[:cutoff].rstrip()


def reset_multimodal_generation_state(model):
    if hasattr(model, "model") and hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None
    elif hasattr(model, "rope_deltas"):
        model.rope_deltas = None


def build_generation_kwargs(inputs, target_input_ids=None, include_mm_token_type_ids=False):
    generation_kwargs = {}
    for key in GENERATION_MEDIA_KEYS:
        if key in inputs:
            generation_kwargs[key] = inputs[key]

    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if include_mm_token_type_ids and mm_token_type_ids is not None:
        if target_input_ids is None or mm_token_type_ids.shape[-1] == target_input_ids.shape[-1]:
            generation_kwargs["mm_token_type_ids"] = mm_token_type_ids
        elif mm_token_type_ids.shape[-1] < target_input_ids.shape[-1]:
            extension_length = target_input_ids.shape[-1] - mm_token_type_ids.shape[-1]
            extension = torch.zeros(
                (mm_token_type_ids.shape[0], extension_length),
                device=mm_token_type_ids.device,
                dtype=mm_token_type_ids.dtype,
            )
            generation_kwargs["mm_token_type_ids"] = torch.cat([mm_token_type_ids, extension], dim=-1)

    return generation_kwargs
