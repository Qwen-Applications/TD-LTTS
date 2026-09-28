"""
Utilities for building MMStar prompts and normalizing answers.
"""
import json
import re

import torch

from methods.latentgrad.generation_utils import GENERATION_MEDIA_KEYS


INLINE_OPTION_PATTERN = re.compile(r"([A-D])\s*:\s*(.*?)(?=(?:,\s*[A-D]\s*:)|$)", flags=re.DOTALL)
LINE_OPTION_PATTERN = re.compile(r"^\(?([A-D])\)?(?:[\.:])?\s+(.+)$")


def _split_mmstar_question_sections(raw_question):
    text = (raw_question or "").strip()
    if not text:
        return None, None, []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    hint = None
    question = None
    choice_lines = []

    for idx, line in enumerate(lines):
        lower = line.lower()
        if lower.startswith("hint:"):
            hint = line.split(":", 1)[1].strip()
            continue
        if lower.startswith("question:"):
            question = line.split(":", 1)[1].strip()
            continue
        if lower in {"choices:", "options:"}:
            choice_lines.extend(lines[idx + 1 :])
            break
        if LINE_OPTION_PATTERN.match(line):
            choice_lines.extend(lines[idx:])
            break

    if question is None:
        prefix_lines = []
        for line in lines:
            lower = line.lower()
            if lower in {"choices:", "options:"} or LINE_OPTION_PATTERN.match(line):
                break
            if lower.startswith("hint:"):
                continue
            prefix_lines.append(line)
        question = " ".join(prefix_lines).strip() or None

    return hint, question, choice_lines


def _parse_mmstar_choices(raw_question):
    text = (raw_question or "").strip()
    hint, question, choice_lines = _split_mmstar_question_sections(text)

    choices = {}
    if "Options:" in text:
        options_blob = text.split("Options:", 1)[1].strip()
        for label, choice_text in INLINE_OPTION_PATTERN.findall(options_blob):
            choices[label] = choice_text.strip().rstrip(",")

    for line in choice_lines:
        match = LINE_OPTION_PATTERN.match(line)
        if match:
            label, choice_text = match.groups()
            choices[label] = choice_text.strip()

    ordered_choices = [choices[label] for label in sorted(choices) if label in choices]
    return hint, question, ordered_choices


def parse_mmstar_question(raw_question):
    hint, question, choices = _parse_mmstar_choices(raw_question)
    question = (question or (raw_question or "").strip()).strip()
    return question, choices, hint


def build_mmstar_question_text(problem):
    stem = problem.get("question", "").strip()
    hint = (problem.get("hint") or "").strip()
    choices = problem.get("choices") or []
    choice_lines = [f"({chr(ord('A') + idx)}) {choice}" for idx, choice in enumerate(choices)]
    lines = []
    if hint:
        lines.append(f"Hint: {hint}")
    lines.extend(
        [
            f"Question: {stem}",
            "Choices:",
            *choice_lines,
            "Answer with the option letter only.",
        ]
    )
    return "\n".join(lines)


def build_mmstar_solver_text(problem, prompt_idx=1):
    question_text = build_mmstar_question_text(problem)
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
        return build_mmstar_answer_only_text(problem)
    raise ValueError(f"Unknown MMStar prompt index: {prompt_idx}")


def build_mmstar_answer_only_text(problem):
    return "\n".join(
        [
            build_mmstar_question_text(problem),
            "Solve the problem from the image carefully.",
            "Return only the final option letter, such as A, B, C, or D.",
            "Do not provide any reasoning or explanation.",
        ]
    )


def mmstar_prompt(problem, image, prompt_idx=1):
    return [
        {
            "role": "system",
            "content": (
                "You are a precise visual question answering system. Analyze the image carefully "
                "and select the correct option."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_mmstar_solver_text(problem, prompt_idx)},
            ],
        },
    ]


def build_mmstar_finalizer_text(problem, draft_response):
    return "\n".join(
        [
            build_mmstar_question_text(problem),
            "Draft solution:",
            draft_response,
            (
                "Extract the final answer from the draft solution. If the draft solution is incomplete, "
                "use the image and question to finish the answer silently before responding."
            ),
            'Return only a JSON object with key "final answer". The "final answer" must be the option letter only.',
        ]
    )


def mmstar_finalizer_prompt(problem, image, draft_response):
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
                {"type": "text", "text": build_mmstar_finalizer_text(problem, draft_response)},
            ],
        },
    ]


def _strip_code_fence(text):
    if text is None:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"<\|im_end\|>|</s>|<\|endoftext\|>", "", text)
    return text.strip()


def extract_mmstar_answer(text, prompt_idx=1):
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


def normalize_mmstar_extraction(extraction, choices, ignore_empty_extractions=False):
    if isinstance(extraction, str):
        extraction = _strip_code_fence(extraction)
    else:
        extraction = "" if extraction is None else str(extraction)

    if ignore_empty_extractions and not extraction:
        return None

    letters = [chr(ord("A") + idx) for idx in range(len(choices or []))]
    if not letters:
        letters = list("ABCD")
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


def find_mmstar_answer_anchor(text, prompt_idx=1):
    if text is None:
        return None
    text = _strip_code_fence(text)
    if prompt_idx == 0:
        match = re.search(r"\\boxed\{", text)
        return match.end() if match else None

    for pattern in [
        r'"final answer"\s*:\s*"?',
        r'"final_answer"\s*:\s*"?',
        r"(?:final answer|final_answer)\s*[:：]\s*\"?",
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.end()
    return None


def estimate_generated_token_offset(tokenizer, text, char_offset):
    if text is None or char_offset is None:
        return None
    prefix = text[:char_offset]
    return len(tokenizer.encode(prefix, add_special_tokens=False))


def resolve_mmstar_start_index(
    output_text,
    tokenizer,
    original_length,
    prompt_idx=1,
    answer_tail_tokens=0,
    fallback_ratio=0.8,
):
    if original_length <= 0:
        return 0

    anchor = find_mmstar_answer_anchor(output_text, prompt_idx=prompt_idx)
    if anchor is not None:
        offset = estimate_generated_token_offset(tokenizer, _strip_code_fence(output_text), anchor)
        if offset is not None:
            return max(0, min(offset - answer_tail_tokens, original_length - 1))

    return max(0, min(int(original_length * fallback_ratio), original_length - 1))


def _score_mmstar_answer_only_letter(model, processor, image, problem, letter):
    prompt_messages = [
        {
            "role": "system",
            "content": (
                "You are a precise visual question answering system. Analyze the image carefully "
                "and select the correct option."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_mmstar_answer_only_text(problem)},
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


def choose_mmstar_pairwise_prediction_by_letter_likelihood(
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

    original_score = _score_mmstar_answer_only_letter(model, processor, image, problem, letter=original_prediction)
    optimized_score = _score_mmstar_answer_only_letter(model, processor, image, problem, letter=optimized_prediction)
    selected = "original" if original_score > optimized_score else "optimized"
    return {
        "selected": selected,
        "original_letter": original_prediction,
        "optimized_letter": optimized_prediction,
        "original_letter_score": original_score,
        "optimized_letter_score": optimized_score,
    }
