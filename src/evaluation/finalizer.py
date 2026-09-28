import torch

from methods.latentgrad.generation_utils import (
    get_stop_token_ids,
    move_model_inputs,
    reset_multimodal_generation_state,
    trim_after_stop_words,
)
from benchmarks.mmstar import build_mmstar_finalizer_text, mmstar_finalizer_prompt
from benchmarks.mathvista import build_mathvista_finalizer_text, mathvista_finalizer_prompt


STOP_WORDS = ["</s>", "<|im_end|>", "<|endoftext|>"]


def should_finalize_answer(
    response_text,
    extraction,
    normalized_prediction,
    data_name,
    prompt_idx,
    mode="when_needed",
):
    if mode == "off":
        return False
    if mode == "always":
        return True

    if normalized_prediction is None:
        return True
    if extraction is None:
        return True

    if ("MathVista" in data_name or "MMStar" in data_name) and prompt_idx == 1:
        lowered = (response_text or "").lower()
        if '"final answer"' not in lowered and '"final_answer"' not in lowered:
            return True

    return False


def finalize_answer(
    model,
    tokenizer,
    processor,
    device,
    data_name,
    prompt_idx,
    prepared,
    draft_response,
    max_new_tokens=64,
):
    if "MathVista" not in data_name and "MMStar" not in data_name:
        return draft_response, None
    if processor is None:
        return draft_response, None

    image = prepared.get("image")
    images = prepared.get("images")
    problem = prepared.get("problem")
    if (image is None and not images) or problem is None:
        return draft_response, None

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if "MMStar" in data_name:
        prompt_text = build_mmstar_finalizer_text(problem, draft_response)
        messages = mmstar_finalizer_prompt(problem, image=image, draft_response=draft_response)
    else:
        prompt_text = build_mathvista_finalizer_text(problem, draft_response)
        messages = mathvista_finalizer_prompt(problem, image=image, draft_response=draft_response)
    formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = processor(images=images or [image], text=[formatted], return_tensors="pt")
    model_inputs = move_model_inputs(model_inputs, device)

    reset_multimodal_generation_state(model)
    with torch.no_grad():
        sequences = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=get_stop_token_ids(tokenizer),
        )

    finalizer_response = tokenizer.decode(
        sequences[0][model_inputs["input_ids"].shape[-1] :],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    stop_words = list(STOP_WORDS)
    eos_token = tokenizer.eos_token
    if eos_token is not None and eos_token not in stop_words:
        stop_words.append(eos_token)
    finalizer_response = trim_after_stop_words(finalizer_response, stop_words)

    metadata = {
        "prompt_text": prompt_text,
        "response": finalizer_response,
    }
    return finalizer_response, metadata
