"""
Utilities for building MathVista prompts and normalizing answers.
"""
import json
import re
from difflib import SequenceMatcher

import torch

from methods.latentgrad.generation_utils import GENERATION_MEDIA_KEYS


def build_mathvista_hint(problem):
    question_type = problem.get("question_type")
    answer_type = problem.get("answer_type")
    precision = problem.get("precision")

    if question_type == "multi_choice":
        return (
            "Hint: Please answer the question and provide the correct option letter, "
            "e.g., A, B, C, D, at the end."
        )
    if answer_type == "integer":
        return (
            "Hint: Please answer the question requiring an integer answer and provide "
            "the final value at the end."
        )
    if answer_type == "float":
        if precision is None:
            return (
                "Hint: Please answer the question requiring a floating-point number and "
                "provide the final value at the end."
            )
        precision = int(precision)
        places = "one" if precision == 1 else "two" if precision == 2 else str(precision)
        return (
            f"Hint: Please answer the question requiring a floating-point number with {places} "
            f"decimal place{'s' if precision != 1 else ''} and provide the final value at the end."
        )
    if answer_type == "list":
        return (
            "Hint: Please answer the question requiring a Python list as an answer and "
            "provide the final list at the end."
        )
    return "Hint: Please provide the final answer at the end."


def build_mathvista_question_text(problem):
    question = problem.get("question", "")
    unit = problem.get("unit")
    choices = problem.get("choices") or []

    question_text = f"Question: {question}"
    if unit:
        question_text += f" (Unit: {unit})"

    choices_text = ""
    if choices:
        lines = ["Choices:"]
        for idx, choice in enumerate(choices):
            lines.append(f"({chr(ord('A') + idx)}) {choice}")
        choices_text = "\n".join(lines)

    return "\n".join(item for item in [question_text, choices_text, build_mathvista_hint(problem)] if item)


def build_mathvista_solver_text(problem, prompt_idx=1):
    prompt_parts = [build_mathvista_question_text(problem)]
    question_type = problem.get("question_type")
    answer_type = problem.get("answer_type")
    precision = problem.get("precision")

    if prompt_idx == 0:
        answer_rule = "put your final answer within \\boxed{}."
        if question_type == "multi_choice":
            answer_rule = "put the final option letter within \\boxed{}."
        prompt_parts.append(f"Please reason step by step, and {answer_rule}")
        return "\n".join(prompt_parts)

    if prompt_idx == 1:
        answer_rule = 'The "final answer" must be the option letter only.'
        if question_type != "multi_choice":
            if answer_type == "float" and precision is not None:
                answer_rule = (
                    f'The "final answer" must be a floating-point number rounded to '
                    f"{int(precision)} decimal place{'s' if int(precision) != 1 else ''}."
                )
            elif answer_type == "integer":
                answer_rule = 'The "final answer" must be an integer.'
            elif answer_type == "list":
                answer_rule = 'The "final answer" must be a Python list.'
            else:
                answer_rule = 'The "final answer" must be the final value only.'

        prompt_parts.extend(
            [
                "Please think step by step.",
                'Please return a JSON object with keys "thought process" and "final answer" only.',
                answer_rule,
            ]
        )
        return "\n".join(prompt_parts)

    if prompt_idx == 2:
        return build_mathvista_answer_only_text(problem)

    raise ValueError(f"Unknown MathVista prompt index: {prompt_idx}")


def build_mathvista_answer_only_text(problem):
    prompt_parts = [build_mathvista_question_text(problem)]
    question_type = problem.get("question_type")
    answer_type = problem.get("answer_type")
    precision = problem.get("precision")

    if question_type == "multi_choice":
        answer_rule = "Return only the final option letter, such as A, B, C, or D."
    elif answer_type == "integer":
        answer_rule = "Return only the final integer."
    elif answer_type == "float" and precision is not None:
        answer_rule = (
            f"Return only the final floating-point number rounded to "
            f"{int(precision)} decimal place{'s' if int(precision) != 1 else ''}."
        )
    elif answer_type == "list":
        answer_rule = "Return only the final Python list."
    else:
        answer_rule = "Return only the final answer."

    prompt_parts.extend(
        [
            "Solve the problem from the image carefully.",
            answer_rule,
            "Do not provide any reasoning or explanation.",
        ]
    )
    return "\n".join(prompt_parts)


def mathvista_prompt(problem, image, prompt_idx=1):
    return [
        {
            "role": "system",
            "content": (
                "You are a precise visual math solver. Analyze the image carefully and "
                "solve the problem."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_mathvista_solver_text(problem, prompt_idx)},
            ],
        },
    ]


def build_mathvista_finalizer_text(problem, draft_response):
    question_text = build_mathvista_question_text(problem)
    question_type = problem.get("question_type")
    answer_type = problem.get("answer_type")
    precision = problem.get("precision")

    answer_rule = 'Return only a JSON object with key "final answer".'
    if question_type == "multi_choice":
        answer_rule += ' The "final answer" must be the option letter only.'
    elif answer_type == "float" and precision is not None:
        answer_rule += (
            f' The "final answer" must be a floating-point number rounded to '
            f"{int(precision)} decimal place{'s' if int(precision) != 1 else ''}."
        )
    elif answer_type == "integer":
        answer_rule += ' The "final answer" must be an integer.'
    elif answer_type == "list":
        answer_rule += ' The "final answer" must be a Python list.'
    else:
        answer_rule += ' The "final answer" must be the final value only.'

    return "\n".join(
        [
            question_text,
            "Draft solution:",
            draft_response,
            (
                "Extract the final answer from the draft solution. If the draft solution is incomplete, "
                "use the image and question to finish the answer silently before responding."
            ),
            answer_rule,
        ]
    )


def mathvista_finalizer_prompt(problem, image, draft_response):
    return [
        {
            "role": "system",
            "content": (
                "You are a precise visual answer finalizer. Read the image and draft solution, "
                "then return only the final answer in the requested JSON format."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_mathvista_finalizer_text(problem, draft_response)},
            ],
        },
    ]


def prediction_to_mathvista_choice_letter(prediction, choices):
    if prediction is None:
        return None
    for idx, choice in enumerate(choices or []):
        if prediction == choice:
            return chr(ord("A") + idx)
    return None


def _score_mathvista_answer_only_letter(model, processor, image, problem, letter):
    prompt_messages = [
        {
            "role": "system",
            "content": (
                "You are a precise visual math solver. Analyze the image carefully and "
                "solve the problem."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_mathvista_answer_only_text(problem)},
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


def choose_mathvista_pairwise_prediction_by_letter_likelihood(
    model,
    processor,
    image,
    problem,
    original_prediction,
    optimized_prediction,
):
    choices = problem.get("choices") or []
    original_letter = prediction_to_mathvista_choice_letter(original_prediction, choices)
    optimized_letter = prediction_to_mathvista_choice_letter(optimized_prediction, choices)
    if original_letter is None or optimized_letter is None:
        return {
            "selected": "optimized",
            "original_letter_score": None,
            "optimized_letter_score": None,
        }

    original_score = _score_mathvista_answer_only_letter(model, processor, image, problem, letter=original_letter)
    optimized_score = _score_mathvista_answer_only_letter(model, processor, image, problem, letter=optimized_letter)
    selected = "original" if original_score > optimized_score else "optimized"
    return {
        "selected": selected,
        "original_letter": original_letter,
        "optimized_letter": optimized_letter,
        "original_letter_score": original_score,
        "optimized_letter_score": optimized_score,
    }


def is_mathvista_answer_available(answer):
    if answer is None:
        return False
    if isinstance(answer, str):
        return answer.strip().lower() not in {"", "none", "null", "nan", "n/a"}
    return True


def _strip_code_fence(text):
    if text is None:
        return None
    text = text.strip()
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
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def extract_mathvista_answer(text, prompt_idx=1):
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
        r"(?:final answer|answer|my answer)\s*[:：]\s*(.+)",
        r"(?:final_answer)\s*[:：]\s*(.+)",
        r'The answer is "(.*)"\.',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip().strip('"').strip("'")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return text.strip() or None


def find_mathvista_answer_anchor(text, prompt_idx=1):
    if text is None:
        return None
    text = _strip_code_fence(text)

    if prompt_idx == 0:
        match = re.search(r"\\boxed\{", text)
        return match.end() if match else None

    patterns = [
        r'"final answer"\s*:\s*"?',
        r'"final_answer"\s*:\s*"?',
        r"(?:final answer|final_answer)\s*[:：]\s*\"?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.end()
    return None


def estimate_generated_token_offset(tokenizer, text, char_offset):
    if text is None or char_offset is None:
        return None
    prefix = text[:char_offset]
    return len(tokenizer.encode(prefix, add_special_tokens=False))


def resolve_mathvista_start_index(
    output_text,
    tokenizer,
    original_length,
    prompt_idx=1,
    answer_tail_tokens=0,
    fallback_ratio=0.8,
):
    if original_length <= 0:
        return 0

    anchor = find_mathvista_answer_anchor(output_text, prompt_idx=prompt_idx)
    if anchor is not None:
        offset = estimate_generated_token_offset(tokenizer, _strip_code_fence(output_text), anchor)
        if offset is not None:
            return max(0, min(offset - answer_tail_tokens, original_length - 1))

    return max(0, min(int(original_length * fallback_ratio), original_length - 1))


def _similarity(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def get_most_similar_choice(extraction, choices):
    return max(choices, key=lambda choice: _similarity(extraction, choice))


def normalize_mathvista_extraction(
    extraction,
    choices,
    question_type,
    answer_type,
    precision,
    ignore_empty_extractions=False,
):
    if question_type == "multi_choice":
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            extraction = "" if extraction is None else str(extraction)
        extraction = _strip_code_fence(extraction)

        if ignore_empty_extractions and not extraction:
            return None
        if extraction in choices:
            return extraction

        compact_extraction = extraction.upper().strip().strip(".").strip(",").strip('"').strip("'")
        direct_choice = re.fullmatch(r"\(?\s*([A-Z])\s*\)?", compact_extraction)
        letters = [chr(ord("A") + idx) for idx in range(len(choices))]
        if direct_choice and direct_choice.group(1) in letters:
            return choices[letters.index(direct_choice.group(1))]

        anchored_letters = re.findall(
            r'(?:FINAL ANSWER|FINAL_ANSWER|ANSWER|OPTION|CHOICE)\s*(?:IS|=|:)?\s*\(?\s*([A-Z])\s*\)?(?:[\s\.,]|$)',
            extraction.upper(),
        )
        if anchored_letters:
            candidate = anchored_letters[-1]
            if candidate in letters:
                return choices[letters.index(candidate)]

        parenthesized_letters = re.findall(r"\(([A-Z])\)", extraction.upper())
        if parenthesized_letters:
            candidate = parenthesized_letters[-1]
            if candidate in letters:
                return choices[letters.index(candidate)]

        for choice in choices:
            if extraction.lower() == str(choice).strip().lower():
                return choice

        direct_letter = re.search(r"\b([A-Z])\b", compact_extraction)
        if direct_letter and direct_letter.group(1) in letters:
            return choices[letters.index(direct_letter.group(1))]

        if len(extraction.split()) <= 8 and extraction:
            best_choice = get_most_similar_choice(extraction, choices)
            if _similarity(extraction, best_choice) >= 0.85:
                return best_choice
        return None

    if answer_type == "integer":
        try:
            cleaned = _strip_code_fence(str(extraction)).replace(",", "").strip()
            return str(int(float(cleaned)))
        except Exception:
            return None

    if answer_type == "float":
        try:
            cleaned = _strip_code_fence(str(extraction)).replace(",", "").strip()
            decimals = int(precision) if precision is not None else 2
            return f"{round(float(cleaned), decimals):.{decimals}f}"
        except Exception:
            return None

    if answer_type == "list":
        try:
            return _strip_code_fence(str(extraction))
        except Exception:
            return None

    return _strip_code_fence(str(extraction)) if extraction is not None else None


def safe_equal(prediction, answer):
    try:
        return prediction == answer
    except Exception:
        return False
