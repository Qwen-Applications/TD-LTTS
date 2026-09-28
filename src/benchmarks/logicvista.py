"""
Utilities for LogicVista prompts and answer normalization.
"""
import json
import re

import torch

from methods.latentgrad.generation_utils import GENERATION_MEDIA_KEYS


def _logicvista_answer_mode(problem):
    problem = problem or {}
    answer = str(problem.get("answer") or "").strip().upper()
    text = str(problem.get("problem") or "")

    # Official LogicVista contains mixed encoding. Prefer the label format when it exists.
    if re.fullmatch(r"[A-I](?:\s*,\s*[A-I])*", answer):
        return "letter"
    if re.fullmatch(r"[1-9](?:\s*,\s*[1-9])*", answer):
        return "number"

    if re.search(r"\b(?:select|answer)\s+from\s+1\s*[-,]\s*5\b", text, flags=re.IGNORECASE):
        return "number"
    if re.search(r"\b1\s*,\s*2\s*,\s*3\s*,\s*4\s*,\s*(?:and\s*)?5\b", text, flags=re.IGNORECASE):
        return "number"
    return "letter"


def build_logicvista_question_text(problem):
    question = (problem.get("problem") or "").replace("<image>", "").strip()
    answer_mode = _logicvista_answer_mode(problem)
    if answer_mode == "number":
        hint = "Hint: Return the final option number only. If multiple choices are correct, return the numbers separated by commas in ascending order."
    else:
        hint = "Hint: Return the final option letter only. If multiple choices are correct, return the letters separated by commas in alphabetical order."
    return "\n".join(
        [
            f"Question: {question}",
            hint,
        ]
    )


def build_logicvista_solver_text(problem, prompt_idx=1):
    prompt_parts = [build_logicvista_question_text(problem)]

    if prompt_idx == 0:
        prompt_parts.append("Please reason step by step, and put the final option letter(s) within \\boxed{}.")
        return "\n".join(prompt_parts)

    if prompt_idx == 1:
        answer_mode = _logicvista_answer_mode(problem)
        final_answer_rule = (
            'The "final answer" must be the option number only. If multiple choices are correct, separate them with commas.'
            if answer_mode == "number"
            else 'The "final answer" must be the option letter only. If multiple choices are correct, separate them with commas.'
        )
        prompt_parts.extend(
            [
                "Please think step by step.",
                'Please return a JSON object with keys "thought process" and "final answer" only.',
                final_answer_rule,
            ]
        )
        return "\n".join(prompt_parts)

    if prompt_idx == 2:
        return build_logicvista_answer_only_text(problem)

    raise ValueError(f"Unknown LogicVista prompt index: {prompt_idx}")


def build_logicvista_answer_only_text(problem):
    prompt_parts = [build_logicvista_question_text(problem)]
    answer_mode = _logicvista_answer_mode(problem)
    answer_rule = (
        "Return only the final option number or comma-separated option numbers."
        if answer_mode == "number"
        else "Return only the final option letter or comma-separated option letters."
    )
    prompt_parts.extend(
        [
            "Solve the visual logic problem carefully.",
            answer_rule,
            "Do not provide any reasoning or explanation.",
        ]
    )
    return "\n".join(prompt_parts)


def logicvista_prompt(problem, image, prompt_idx=1):
    return [
        {
            "role": "system",
            "content": "You are a precise visual reasoning solver. Analyze the image carefully and solve the pattern or logic problem.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_logicvista_solver_text(problem, prompt_idx)},
            ],
        },
    ]


def _score_logicvista_answer_only_prediction(model, processor, image, problem, prediction):
    prompt_messages = [
        {
            "role": "system",
            "content": "You are a precise visual reasoning solver. Analyze the image carefully and solve the pattern or logic problem.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_logicvista_answer_only_text(problem)},
            ],
        },
    ]
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_batch = processor(images=[image], text=[prompt_text], return_tensors="pt")
    prompt_batch = {
        key: value.to(model.device) if torch.is_tensor(value) else value
        for key, value in prompt_batch.items()
    }
    cand_ids = processor.tokenizer(str(prediction), add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)

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


def choose_logicvista_pairwise_prediction_likelihood(model, processor, image, problem, original_prediction, optimized_prediction):
    original_score = _score_logicvista_answer_only_prediction(
        model=model,
        processor=processor,
        image=image,
        problem=problem,
        prediction=original_prediction,
    )
    optimized_score = _score_logicvista_answer_only_prediction(
        model=model,
        processor=processor,
        image=image,
        problem=problem,
        prediction=optimized_prediction,
    )
    selected = "original" if original_score > optimized_score else "optimized"
    return {
        "selected": selected,
        "original_score": original_score,
        "optimized_score": optimized_score,
    }


def extract_logicvista_answer(text, prompt_idx=1):
    if text is None:
        return None

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ["final answer", "final_answer", "answer", "my answer"]:
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
        r"(?:final answer|answer|my answer)\s*[:：]\s*(.+)",
        r"(?:final_answer)\s*[:：]\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip().strip('"').strip("'")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return text.strip() or None


def normalize_logicvista_extraction(extraction, ignore_empty_extractions=False, problem=None):
    if extraction is None:
        return None

    text = str(extraction).strip().upper()
    if ignore_empty_extractions and not text:
        return None

    answer_mode = _logicvista_answer_mode(problem)
    if answer_mode == "number":
        matches = re.findall(r"\b([1-9])\b", text)
        if not matches:
            matches = re.findall(r"[1-9]", text)
    else:
        matches = re.findall(r"\b([A-I])\b", text)
        if not matches:
            matches = re.findall(r"[A-I]", text)
    if not matches:
        return None

    ordered = []
    for token in matches:
        if token not in ordered:
            ordered.append(token)
    return ", ".join(ordered)


def is_logicvista_answer_available(problem):
    if problem is None:
        return False
    answer = problem.get("answer")
    return normalize_logicvista_extraction(answer, problem=problem) is not None
