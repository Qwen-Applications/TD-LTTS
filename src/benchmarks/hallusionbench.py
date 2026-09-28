"""
Utilities for HallusionBench prompts, answer normalization, and scoring.
"""
import json
import re

import torch

from methods.latentgrad.generation_utils import GENERATION_MEDIA_KEYS


YES_SET = {"yes", "y", "true", "1"}
NO_SET = {"no", "n", "false", "0"}


def hallusionbench_label_to_text(label):
    if label is None:
        return None
    text = str(label).strip().lower()
    if text in YES_SET:
        return "Yes"
    if text in NO_SET:
        return "No"
    return None


def build_hallusionbench_question_text(problem):
    question = (problem.get("question") or "").strip()
    lines = [f"Question: {question}", "Answer with yes or no only."]
    if problem.get("visual_input") == "0":
        lines.append("This sample has no image input. Answer using the question only.")
    return "\n".join(lines)


def build_hallusionbench_solver_text(problem, prompt_idx=1):
    question_text = build_hallusionbench_question_text(problem)
    if prompt_idx == 0:
        return "\n".join(
            [
                question_text,
                "Please reason step by step, and put the final answer within \\boxed{}.",
            ]
        )
    if prompt_idx == 1:
        return "\n".join(
            [
                question_text,
                "Please think step by step.",
                'Please return a JSON object with keys "thought process" and "final answer" only.',
                'The "final answer" must be either "yes" or "no".',
            ]
        )
    if prompt_idx == 2:
        return build_hallusionbench_answer_only_text(problem)
    raise ValueError(f"Unknown HallusionBench prompt index: {prompt_idx}")


def build_hallusionbench_answer_only_text(problem):
    return "\n".join(
        [
            build_hallusionbench_question_text(problem),
            'Return only "yes" or "no".',
            "Do not provide any reasoning or explanation.",
        ]
    )


def hallusionbench_prompt(problem, image=None, prompt_idx=1):
    user_content = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": build_hallusionbench_solver_text(problem, prompt_idx)})
    return [
        {
            "role": "system",
            "content": (
                "You are a precise visual reasoning system. Read the question carefully and answer yes or no."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def _strip_code_fence(text):
    if text is None:
        return None
    text = str(text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"<\|im_end\|>|</s>|<\|endoftext\|>", "", text)
    return text.strip()


def extract_hallusionbench_answer(text, prompt_idx=1):
    if text is None:
        return None

    text = _strip_code_fence(text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ["final answer", "final_answer", "answer"]:
                if key in obj:
                    value = obj[key]
                    return str(value).strip() if value is not None else None
    except Exception:
        pass

    if prompt_idx == 0:
        boxed_matches = list(re.finditer(r"\\boxed\{(.*?)\}", text, re.DOTALL))
        if boxed_matches:
            return boxed_matches[-1].group(1).strip()

    patterns = [
        r'"final answer"\s*:\s*"?(.*?)"?(?:[,}\n]|$)',
        r'"final_answer"\s*:\s*"?(.*?)"?(?:[,}\n]|$)',
        r"(?:final answer|final_answer|answer)\s*(?:is|=|:)?\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip().strip('"').strip("'")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return text.strip() or None


def normalize_hallusionbench_extraction(extraction, ignore_empty_extractions=False):
    text = _strip_code_fence(extraction)
    if text is None:
        return None
    if ignore_empty_extractions and not text:
        return None

    compact = text.strip().lower()
    compact = re.sub(r"^[\s\.,:;\"'`()\[\]\-]+|[\s\.,:;\"'`()\[\]\-]+$", "", compact)

    direct = re.search(r"\b(yes|no|true|false)\b", compact)
    if direct:
        token = direct.group(1)
        return "Yes" if token in {"yes", "true"} else "No"

    if compact in YES_SET:
        return "Yes"
    if compact in NO_SET:
        return "No"
    return None


def is_hallusionbench_answer_available(answer):
    return hallusionbench_label_to_text(answer) is not None


def _score_hallusionbench_answer_only_prediction(model, processor, media, problem, prediction):
    prompt_messages = [
        {
            "role": "system",
            "content": "You are a precise visual reasoning system. Read the question carefully and answer yes or no.",
        },
        {
            "role": "user",
            "content": [],
        },
    ]
    images = None
    image = None
    if media is not None:
        images = media.get("images")
        image = media.get("image")
        if image is None and images:
            image = images[0]
    if problem.get("visual_input") == "1" and image is not None:
        prompt_messages[1]["content"].append({"type": "image", "image": image})
    prompt_messages[1]["content"].append({"type": "text", "text": build_hallusionbench_answer_only_text(problem)})

    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    if problem.get("visual_input") == "1" and image is not None:
        prompt_batch = processor(images=images or [image], text=[prompt_text], return_tensors="pt")
    else:
        prompt_batch = processor(text=[prompt_text], return_tensors="pt")
    prompt_batch = {
        key: value.to(model.device) if torch.is_tensor(value) else value
        for key, value in prompt_batch.items()
    }

    cand_ids = processor.tokenizer(str(prediction), add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    if cand_ids.shape[1] == 0:
        return float("-inf")

    input_ids = torch.cat([prompt_batch["input_ids"], cand_ids], dim=1)
    attention_mask = torch.cat([prompt_batch["attention_mask"], torch.ones_like(cand_ids)], dim=1)
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    for key in GENERATION_MEDIA_KEYS:
        if key in prompt_batch:
            model_inputs[key] = prompt_batch[key]

    with torch.no_grad():
        outputs = model(**model_inputs)
        prompt_len = prompt_batch["input_ids"].shape[1]
        logits = outputs.logits[:, prompt_len - 1 : -1, :]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        token_logprobs = logprobs.gather(-1, cand_ids.unsqueeze(-1)).squeeze(-1)
    return token_logprobs.mean().item()


def choose_hallusionbench_pairwise_prediction_likelihood(
    model,
    processor,
    media,
    problem,
    original_prediction,
    optimized_prediction,
):
    original_score = _score_hallusionbench_answer_only_prediction(
        model=model,
        processor=processor,
        media=media,
        problem=problem,
        prediction=original_prediction,
    )
    optimized_score = _score_hallusionbench_answer_only_prediction(
        model=model,
        processor=processor,
        media=media,
        problem=problem,
        prediction=optimized_prediction,
    )
    selected = "original" if original_score > optimized_score else "optimized"
    return {
        "selected": selected,
        "original_score": original_score,
        "optimized_score": optimized_score,
    }
