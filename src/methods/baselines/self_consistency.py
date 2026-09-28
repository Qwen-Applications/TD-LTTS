from collections import OrderedDict

import torch

from evaluation.extract_judge_answer import extract_answer, normalize_prediction
from methods.latentgrad.generation_utils import (
    get_stop_token_ids,
    move_model_inputs,
    reset_multimodal_generation_state,
    trim_after_stop_words,
)

STOP_WORDS = ["</s>", "<|im_end|>", "<|endoftext|>"]


def _sample_answer(
    input_text,
    model,
    tokenizer,
    device,
    prompt_inputs=None,
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.95,
):
    eos_token = tokenizer.eos_token
    if eos_token is not None and eos_token not in STOP_WORDS:
        STOP_WORDS.append(eos_token)

    if prompt_inputs is None:
        inputs = tokenizer([input_text], return_tensors="pt").to(device)
    else:
        inputs = move_model_inputs(prompt_inputs, device)

    prompt_length = inputs["input_ids"].shape[-1]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    reset_multimodal_generation_state(model)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=max(float(temperature), 1e-5),
            top_p=min(max(float(top_p), 1e-5), 1.0),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=get_stop_token_ids(tokenizer),
        )

    answer = tokenizer.decode(
        generated[0][prompt_length:],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return trim_after_stop_words(answer, STOP_WORDS)


def _build_candidate_record(answer_text, data_name, prompt_idx, model_name_or_path, problem):
    extraction = extract_answer(
        answer_text,
        data_name=data_name,
        prompt_idx=prompt_idx,
        model_name=model_name_or_path,
        problem=problem,
    )
    prediction = normalize_prediction(extraction, data_name=data_name, problem=problem) if extraction is not None else None
    return {
        "answer_text": answer_text,
        "extraction": extraction,
        "prediction": prediction,
    }


def self_consistency_generation(
    model,
    tokenizer,
    device,
    input_text,
    prompt_inputs,
    max_new_tokens,
    data_name,
    prompt_idx,
    model_name_or_path,
    problem,
    num_candidates=4,
    temperature=0.7,
    top_p=0.95,
):
    candidates = []
    for _ in range(max(int(num_candidates), 1)):
        answer_text = _sample_answer(
            input_text=input_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_inputs=prompt_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        candidates.append(
            _build_candidate_record(
                answer_text=answer_text,
                data_name=data_name,
                prompt_idx=prompt_idx,
                model_name_or_path=model_name_or_path,
                problem=problem,
            )
        )

    valid_groups = OrderedDict()
    for idx, item in enumerate(candidates):
        prediction = item["prediction"]
        if prediction is None:
            continue
        key = str(prediction).strip().lower()
        if key not in valid_groups:
            valid_groups[key] = {"count": 0, "first_idx": idx, "best_idx": idx}
        valid_groups[key]["count"] += 1

    if valid_groups:
        best_key = None
        best_count = -1
        best_first_idx = len(candidates)
        for key, meta in valid_groups.items():
            if meta["count"] > best_count or (meta["count"] == best_count and meta["first_idx"] < best_first_idx):
                best_key = key
                best_count = meta["count"]
                best_first_idx = meta["first_idx"]
        selected_idx = valid_groups[best_key]["best_idx"]
    else:
        selected_idx = 0

    selected = candidates[selected_idx]
    baseline_info = {
        "mode": "self_consistency",
        "num_candidates": len(candidates),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "selected_index": selected_idx,
        "candidates": candidates,
    }
    return selected["answer_text"], [0.0], baseline_info
