"""
Utilities for RealWorldQA prompt construction, answer extraction, and normalization.
"""
import json
import re

import torch

from methods.latentgrad.generation_utils import GENERATION_MEDIA_KEYS


OPTION_LINE_PATTERN = re.compile(r"^\s*([A-I])(?:[\.\):]\s*|\s+)(.+?)\s*$")


def parse_realworldqa_question(raw_question):
    text = (raw_question or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    stem_lines = []
    choice_map = {}
    instructions = []
    option_started = False

    for line in lines:
        match = OPTION_LINE_PATTERN.match(line)
        if match:
            option_started = True
            label, choice_text = match.groups()
            choice_map[label] = choice_text.strip()
            continue

        lower = line.lower()
        if lower.startswith("please answer directly"):
            instructions.append(line)
            continue

        if option_started:
            instructions.append(line)
        else:
            stem_lines.append(line)

    stem = "\n".join(stem_lines).strip()
    choices = [choice_map[k] for k in sorted(choice_map)]
    hint = " ".join(instructions).strip() or None
    question_type = "multi_choice" if choices else "open"
    return stem, choices, hint, question_type


def build_realworldqa_question_text(problem):
    stem = (problem.get("question") or "").strip()
    hint = (problem.get("hint") or "").strip()
    choices = problem.get("choices") or []
    lines = [f"Question: {stem}"]
    if choices:
        lines.append("Choices:")
        lines.extend(f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices))
        lines.append("Answer with the option letter only.")
    else:
        lines.append("Answer with a short direct answer only.")
    if hint:
        lines.append(f"Original instruction: {hint}")
    return "\n".join(lines)


def build_realworldqa_solver_text(problem, prompt_idx=1):
    question_text = build_realworldqa_question_text(problem)
    if prompt_idx == 0:
        return "\n".join(
            [
                question_text,
                "Please reason step by step, and put the final answer within \\boxed{}.",
            ]
        )
    if prompt_idx == 1:
        final_rule = (
            'The "final answer" must be the option letter only.'
            if problem.get("question_type") == "multi_choice"
            else 'The "final answer" must be a short direct answer only.'
        )
        return "\n".join(
            [
                question_text,
                "Please think step by step.",
                'Please return a JSON object with keys "thought process" and "final answer" only.',
                final_rule,
            ]
        )
    if prompt_idx == 2:
        return build_realworldqa_answer_only_text(problem)
    raise ValueError(f"Unknown RealWorldQA prompt index: {prompt_idx}")


def build_realworldqa_answer_only_text(problem):
    if problem.get("question_type") == "multi_choice":
        answer_rule = "Return only the final option letter, such as A, B, C, or D."
    else:
        answer_rule = "Return only the final short answer."
    return "\n".join(
        [
            build_realworldqa_question_text(problem),
            "Solve the real-world visual question carefully.",
            answer_rule,
            "Do not provide any reasoning or explanation.",
        ]
    )


def realworldqa_prompt(problem, image, prompt_idx=1):
    return [
        {
            "role": "system",
            "content": "You are a precise visual question answering system. Analyze the image carefully and answer the question.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_realworldqa_solver_text(problem, prompt_idx)},
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


def extract_realworldqa_answer(text, prompt_idx=1):
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


def _normalize_freeform_answer(text):
    text = _strip_code_fence(text)
    if text is None:
        return None
    text = text.strip().casefold()
    text = re.sub(r"^[\s\.,:;\"'`()\[\]\-]+|[\s\.,:;\"'`()\[\]\-]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text or None


def normalize_realworldqa_extraction(extraction, problem, ignore_empty_extractions=False):
    question_type = problem.get("question_type")
    choices = problem.get("choices") or []

    if question_type == "multi_choice":
        text = _strip_code_fence(extraction)
        if text is None:
            return None
        text = text.upper().strip().strip(".").strip(",").strip('"').strip("'")
        letters = [chr(ord("A") + idx) for idx in range(len(choices))]
        direct = re.fullmatch(r"\(?\s*([A-I])\s*\)?", text)
        if direct and direct.group(1) in letters:
            return direct.group(1)
        anchored = re.findall(
            r"(?:FINAL ANSWER|FINAL_ANSWER|ANSWER|OPTION|CHOICE)\s*(?:IS|=|:)?\s*\(?\s*([A-I])\s*\)?(?:[\s\.,]|$)",
            text,
        )
        if anchored:
            candidate = anchored[-1]
            if candidate in letters:
                return candidate
        for idx, choice in enumerate(choices):
            if text.casefold() == str(choice).strip().casefold():
                return letters[idx]
        return None if ignore_empty_extractions and not text else None

    normalized = _normalize_freeform_answer(extraction)
    if ignore_empty_extractions and not normalized:
        return None
    return normalized


def is_realworldqa_answer_available(answer):
    return _normalize_freeform_answer(answer) is not None


def _score_realworldqa_answer_only_prediction(model, processor, image, problem, prediction):
    prompt_messages = [
        {
            "role": "system",
            "content": "You are a precise visual question answering system. Analyze the image carefully and answer the question.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_realworldqa_answer_only_text(problem)},
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


def choose_realworldqa_pairwise_prediction_likelihood(model, processor, image, problem, original_prediction, optimized_prediction):
    original_score = _score_realworldqa_answer_only_prediction(
        model=model,
        processor=processor,
        image=image,
        problem=problem,
        prediction=original_prediction,
    )
    optimized_score = _score_realworldqa_answer_only_prediction(
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
