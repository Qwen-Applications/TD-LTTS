"""Utilities for MMMU prompts, answer normalization, and single-image scoring."""

import ast
import json
import re
from difflib import SequenceMatcher

import torch

from methods.latentgrad.generation_utils import GENERATION_MEDIA_KEYS


IMAGE_TAG_PATTERN = re.compile(r"<image\s*\d+>", flags=re.IGNORECASE)


def parse_mmmu_options(options):
    if options is None:
        return []
    if isinstance(options, (list, tuple)):
        return [str(item) for item in options]
    text = str(options).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return []


def clean_mmmu_question(question):
    text = re.sub(r"^\s*<image\s*\d+>\s*", "In the image, ", str(question or ""), flags=re.IGNORECASE)
    text = IMAGE_TAG_PATTERN.sub("the image", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_mmmu_question_text(problem):
    stem = clean_mmmu_question(problem.get("question"))
    choices = problem.get("choices") or []
    question_type = problem.get("question_type")

    lines = [f"Question: {stem}"]
    if choices:
        lines.append("Choices:")
        lines.extend(f"({chr(ord('A') + idx)}) {choice}" for idx, choice in enumerate(choices))
    if question_type == "multi_choice":
        lines.append("Answer with the option letter only.")
    else:
        lines.append("Answer with the final value or short answer only.")
    return "\n".join(lines)


def build_mmmu_solver_text(problem, prompt_idx=1):
    question_text = build_mmmu_question_text(problem)
    if prompt_idx == 0:
        answer_rule = "put your final answer within \\boxed{}."
        if problem.get("question_type") == "multi_choice":
            answer_rule = "put the final option letter within \\boxed{}."
        return "\n".join([question_text, f"Please reason step by step, and {answer_rule}"])
    if prompt_idx == 1:
        if problem.get("question_type") == "multi_choice":
            answer_rule = 'The "final answer" must be the option letter only.'
        else:
            answer_rule = 'The "final answer" must be the final value or short answer only.'
        return "\n".join(
            [
                question_text,
                "Please think step by step.",
                'Please return a JSON object with keys "thought process" and "final answer" only.',
                answer_rule,
            ]
        )
    raise ValueError(f"Unknown MMMU prompt index: {prompt_idx}")


def build_mmmu_answer_only_text(problem):
    if problem.get("question_type") == "multi_choice":
        answer_rule = "Return only the final option letter, such as A, B, C, or D."
    else:
        answer_rule = "Return only the final value or short answer."
    return "\n".join(
        [
            build_mmmu_question_text(problem),
            "Solve the problem from the image carefully.",
            answer_rule,
            "Do not provide any reasoning or explanation.",
        ]
    )


def mmmu_prompt(problem, image, prompt_idx=1):
    return [
        {
            "role": "system",
            "content": (
                "You are a precise multidisciplinary visual question answering system. "
                "Analyze the image carefully and solve the problem."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_mmmu_solver_text(problem, prompt_idx)},
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


def _extract_json_object(text):
    if text is None:
        return None
    text = _strip_code_fence(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def extract_mmmu_answer(text, prompt_idx=1):
    if text is None:
        return None
    text = _strip_code_fence(text)
    json_obj = _extract_json_object(text)
    if isinstance(json_obj, dict):
        for key in ["final answer", "final_answer", "answer", "my answer"]:
            if key in json_obj:
                value = json_obj[key]
                return str(value).strip() if value is not None else None

    if prompt_idx == 0:
        boxed_matches = list(re.finditer(r"\\boxed\{(.*?)\}", text, re.DOTALL))
        if boxed_matches:
            return boxed_matches[-1].group(1).strip()

    patterns = [
        r'"final answer"\s*:\s*"?(.*?)"?(?:[,}\n]|$)',
        r'"final_answer"\s*:\s*"?(.*?)"?(?:[,}\n]|$)',
        r"(?:final answer|final_answer|answer|my answer)\s*[:：]\s*(.+)",
        r"The answer is\s+(.+?)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip().strip('"').strip("'")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return text.strip() or None


def _choice_similarity(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def _normalize_open_answer(value):
    if value is None:
        return None
    text = _strip_code_fence(str(value))
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"\s+", " ", text)
    return text if text else None


def normalize_mmmu_extraction(extraction, problem=None, ignore_empty_extractions=False):
    problem = problem or {}
    choices = problem.get("choices") or []
    question_type = problem.get("question_type")

    if question_type == "multi_choice":
        extraction = "" if extraction is None else _strip_code_fence(str(extraction))
        if ignore_empty_extractions and not extraction:
            return None

        letters = [chr(ord("A") + idx) for idx in range(len(choices))]
        if not letters:
            letters = list("ABCDEFGHI")
        compact = extraction.upper().strip().strip(".").strip(",").strip('"').strip("'")

        direct = re.fullmatch(r"\(?\s*([A-Z])\s*\)?", compact)
        if direct and direct.group(1) in letters:
            return direct.group(1)

        anchored = re.findall(
            r"(?:FINAL ANSWER|FINAL_ANSWER|ANSWER|OPTION|CHOICE)\s*(?:IS|=|:)?\s*\(?\s*([A-Z])\s*\)?(?:[\s\.,]|$)",
            compact,
        )
        if anchored and anchored[-1] in letters:
            return anchored[-1]

        parenthesized = re.findall(r"\(([A-Z])\)", compact)
        if parenthesized and parenthesized[-1] in letters:
            return parenthesized[-1]

        for idx, choice in enumerate(choices):
            if extraction.strip().lower() == str(choice).strip().lower():
                return chr(ord("A") + idx)

        if len(extraction.split()) <= 12 and choices:
            best_idx = max(range(len(choices)), key=lambda idx: _choice_similarity(extraction, choices[idx]))
            if _choice_similarity(extraction, choices[best_idx]) >= 0.85:
                return chr(ord("A") + best_idx)
        return None

    normalized = _normalize_open_answer(extraction)
    if ignore_empty_extractions and not normalized:
        return None
    return normalized


def mmmu_answers_equal(prediction, answer, problem=None):
    problem = problem or {}
    if problem.get("question_type") == "multi_choice":
        return str(prediction).strip().upper() == str(answer).strip().upper()

    pred = _normalize_open_answer(prediction)
    label = _normalize_open_answer(answer)
    if pred is None or label is None:
        return pred == label
    if pred.lower() == label.lower():
        return True

    pred_num = pred.replace(",", "")
    label_num = label.replace(",", "")
    try:
        return abs(float(pred_num) - float(label_num)) <= 1e-6
    except Exception:
        return False


def _score_mmmu_answer_only_letter(model, processor, image, problem, letter):
    prompt_messages = [
        {
            "role": "system",
            "content": (
                "You are a precise multidisciplinary visual question answering system. "
                "Analyze the image carefully and select the correct option."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_mmmu_answer_only_text(problem)},
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


def choose_mmmu_pairwise_prediction_by_letter_likelihood(
    model,
    processor,
    image,
    problem,
    original_prediction,
    optimized_prediction,
):
    if original_prediction is None or optimized_prediction is None:
        return {
            "selected": "optimized",
            "original_letter_score": None,
            "optimized_letter_score": None,
        }

    original_score = _score_mmmu_answer_only_letter(model, processor, image, problem, letter=original_prediction)
    optimized_score = _score_mmmu_answer_only_letter(model, processor, image, problem, letter=optimized_prediction)
    selected = "original" if original_score > optimized_score else "optimized"
    return {
        "selected": selected,
        "original_letter": original_prediction,
        "optimized_letter": optimized_prediction,
        "original_letter_score": original_score,
        "optimized_letter_score": optimized_score,
    }
