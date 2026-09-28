"""
Utilities for generating and formatting visual clues.
"""

import json
import re

import torch

from methods.latentgrad.generation_utils import get_stop_token_ids, reset_multimodal_generation_state, trim_after_stop_words
from prompts.clue_prompts import build_visual_clue_request_messages, build_visual_clue_support_messages


STOP_WORDS = ["</s>", "<|im_end|>", "<|endoftext|>"]


def _strip_code_fence(text):
    if text is None:
        return ""
    text = str(text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _normalize_clue_text(text):
    text = str(text or "").strip()
    text = re.sub(r"^\s*(?:[-*•]|\d+[\.\)])\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_visual_clues(text, num_clues=3):
    text = _strip_code_fence(text)
    text = re.sub(r"<\|im_end\|>|</s>|<\|endoftext\|>", "", text).strip()
    if not text:
        return []

    json_candidate = text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_candidate = text[start : end + 1]
    try:
        obj = json.loads(json_candidate)
        if isinstance(obj, dict):
            for key in ["clues", "visual_clues", "facts", "evidence"]:
                value = obj.get(key)
                if isinstance(value, list):
                    text = "\n".join(str(item) for item in value if item is not None)
                    break
        elif isinstance(obj, list):
            text = "\n".join(str(item) for item in obj if item is not None)
    except Exception:
        pass

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1 and ";" in text:
        lines = [item.strip() for item in text.split(";") if item.strip()]

    clues = []
    seen = set()
    for line in lines:
        clue = _normalize_clue_text(line)
        if not clue:
            continue
        key = clue.lower()
        if key in seen:
            continue
        seen.add(key)
        clues.append(clue)
        if len(clues) >= int(num_clues):
            break
    return clues


def format_visual_clues(clues):
    formatted = []
    for clue in clues or []:
        clean = _normalize_clue_text(clue)
        if clean:
            formatted.append(f"- {clean}")
    return "\n".join(formatted)


def generate_visual_clues(
    model,
    tokenizer,
    processor,
    device,
    problem,
    media=None,
    draft_response=None,
    max_new_tokens=96,
    num_clues=3,
):
    if processor is None or not media:
        return []

    messages = build_visual_clue_request_messages(
        problem=problem,
        media=media,
        num_clues=num_clues,
        draft_response=draft_response,
        include_image=True,
    )
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    images = media.get("images") if media else None
    image = media.get("image") if media else None
    if image is None and images:
        image = images[0]
    if image is None and not images:
        return []

    batch = processor(images=images or [image], text=[prompt_text], return_tensors="pt")
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    stop_token_ids = get_stop_token_ids(tokenizer)
    reset_multimodal_generation_state(model)
    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids,
        )

    prompt_len = batch["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_len:]
    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    eos_token = tokenizer.eos_token
    stop_words = list(STOP_WORDS)
    if eos_token and eos_token not in stop_words:
        stop_words.append(eos_token)
    text = trim_after_stop_words(text, stop_words)
    return parse_visual_clues(text, num_clues=num_clues)


def generate_candidate_visual_clues(
    model,
    tokenizer,
    processor,
    device,
    problem,
    media=None,
    candidate_answer=None,
    max_new_tokens=64,
    num_clues=3,
):
    if processor is None or not media:
        return []

    messages = build_visual_clue_support_messages(
        problem=problem,
        media=media,
        candidate_answer=candidate_answer,
        num_clues=num_clues,
        include_image=True,
    )
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    images = media.get("images") if media else None
    image = media.get("image") if media else None
    if image is None and images:
        image = images[0]
    if image is None and not images:
        return []

    batch = processor(images=images or [image], text=[prompt_text], return_tensors="pt")
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    stop_token_ids = get_stop_token_ids(tokenizer)
    reset_multimodal_generation_state(model)
    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids,
        )

    prompt_len = batch["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_len:]
    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    eos_token = tokenizer.eos_token
    stop_words = list(STOP_WORDS)
    if eos_token and eos_token not in stop_words:
        stop_words.append(eos_token)
    text = trim_after_stop_words(text, stop_words)
    return parse_visual_clues(text, num_clues=num_clues)
