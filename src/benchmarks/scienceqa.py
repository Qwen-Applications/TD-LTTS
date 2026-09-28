"""
Utilities for ScienceQA-IMG prompts and answer normalization.
"""
import json
import re

import torch

from methods.latentgrad.generation_utils import GENERATION_MEDIA_KEYS


def answer_index_to_letter(answer_idx, choices):
    if answer_idx is None:
        return None
    try:
        idx = int(answer_idx)
    except Exception:
        return None
    if idx < 0 or idx >= len(choices or []):
        return None
    return chr(ord("A") + idx)


def build_scienceqa_question_text(problem):
    stem = (problem.get("question") or "").strip()
    hint = (problem.get("hint") or "").strip()
    lecture = (problem.get("lecture") or "").strip()
    choices = problem.get("choices") or []
    lines = []
    if hint:
        lines.append(f"Hint: {hint}")
    lines.append(f"Question: {stem}")
    if choices:
        lines.append("Choices:")
        lines.extend(f"({chr(ord('A') + idx)}) {choice}" for idx, choice in enumerate(choices))
    lines.append("Answer with the option letter only.")
    if lecture:
        lines.append(f"Background: {lecture}")
    return "\n".join(lines)


def build_scienceqa_solver_text(problem, prompt_idx=1):
    question_text = build_scienceqa_question_text(problem)
    if prompt_idx == 0:
        return "\n".join(
            [
                question_text,
                "Please reason step by step, and put the final option letter within \\boxed{}.",
            ]
        )
    if prompt_idx == 1:
        return "\n".join(
            [
                question_text,
                "Please think step by step.",
                'Please return a JSON object with keys "thought process" and "final answer" only.',
                'The "final answer" must be the option letter only.',
            ]
        )
    if prompt_idx == 2:
        return build_scienceqa_answer_only_text(problem)
    raise ValueError(f"Unknown ScienceQA prompt index: {prompt_idx}")


def build_scienceqa_answer_only_text(problem):
    return "\n".join(
        [
            build_scienceqa_question_text(problem),
            "Solve the science question from the image carefully.",
            "Return only the final option letter, such as A, B, C, or D.",
            "Do not provide any reasoning or explanation.",
        ]
    )


def scienceqa_prompt(problem, image, prompt_idx=1):
    return [
        {
            "role": "system",
            "content": (
                "You are a precise visual science question answering system. Analyze the image carefully "
                "and select the correct option."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_scienceqa_solver_text(problem, prompt_idx)},
            ],
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


def extract_scienceqa_answer(text, prompt_idx=1):
    if text is None:
        return None

    text = _strip_code_fence(text)
    try:
        answer = json.loads(text)
        if isinstance(answer, dict):
            for key in ["final answer", "final_answer", "answer"]:
                if key in answer:
                    return str(answer[key]).strip()
    except Exception:
        pass

    patterns = [
        r'"final answer"\s*:\s*"?(.*?)"?(?:[,}\n]|$)',
        r'"final_answer"\s*:\s*"?(.*?)"?(?:[,}\n]|$)',
        r"(?:final answer|final_answer|answer|option|choice)\s*(?:is|=|:)?\s*\(?\s*([A-Z])\s*\)?",
        r"\(([A-Z])\)",
        r"\b([A-Z])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            candidate = matches[-1]
            if isinstance(candidate, tuple):
                candidate = candidate[-1]
            return str(candidate).strip()
    return None


def normalize_scienceqa_extraction(extraction, choices, ignore_empty_extractions=False):
    if isinstance(extraction, str):
        extraction = _strip_code_fence(extraction)
    else:
        extraction = "" if extraction is None else str(extraction)

    if ignore_empty_extractions and not extraction:
        return None

    letters = [chr(ord("A") + idx) for idx in range(len(choices or []))]
    compact = extraction.upper().strip().strip(".").strip(",").strip('"').strip("'")

    direct = re.fullmatch(r"\(?\s*([A-Z])\s*\)?", compact)
    if direct and direct.group(1) in letters:
        return direct.group(1)

    anchored = re.findall(
        r"(?:FINAL ANSWER|FINAL_ANSWER|ANSWER|OPTION|CHOICE)\s*(?:IS|=|:)?\s*\(?\s*([A-Z])\s*\)?(?:[\s\.,]|$)",
        compact,
    )
    if anchored:
        candidate = anchored[-1]
        if candidate in letters:
            return candidate

    for idx, choice in enumerate(choices or []):
        if compact == str(choice).strip().upper():
            return letters[idx]

    return None


def _score_scienceqa_answer_only_letter(model, processor, image, problem, letter):
    prompt_messages = [
        {
            "role": "system",
            "content": (
                "You are a precise visual science question answering system. Analyze the image carefully "
                "and select the correct option."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_scienceqa_answer_only_text(problem)},
            ],
        },
    ]
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_batch = processor(images=[image], text=[prompt_text], return_tensors="pt")
    prompt_batch = {
        key: value.to(model.device) if torch.is_tensor(value) else value
        for key, value in prompt_batch.items()
    }
    cand_ids = processor.tokenizer(letter, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)

    input_ids = torch.cat([prompt_batch["input_ids"], cand_ids], dim=1)
    attention_mask = torch.cat([prompt_batch["attention_mask"], torch.ones_like(cand_ids)], dim=1)
    model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
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


def choose_scienceqa_pairwise_prediction_by_letter_likelihood(
    model,
    processor,
    image,
    problem,
    original_prediction,
    optimized_prediction,
):
    choices = problem.get("choices") or []
    original_letter = original_prediction
    optimized_letter = optimized_prediction
    if original_letter is None or optimized_letter is None:
        return {
            "selected": "optimized",
            "original_letter_score": None,
            "optimized_letter_score": None,
        }

    original_score = _score_scienceqa_answer_only_letter(model, processor, image, problem, letter=original_letter)
    optimized_score = _score_scienceqa_answer_only_letter(model, processor, image, problem, letter=optimized_letter)
    selected = "original" if original_score > optimized_score else "optimized"
    return {
        "selected": selected,
        "original_letter_score": original_score,
        "optimized_letter_score": optimized_score,
    }
