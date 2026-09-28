import json
import re

from benchmarks.hallusionbench import extract_hallusionbench_answer, hallusionbench_label_to_text, normalize_hallusionbench_extraction
from benchmarks.logicvista import extract_logicvista_answer, normalize_logicvista_extraction
from benchmarks.mmstar import extract_mmstar_answer, normalize_mmstar_extraction
from benchmarks.mmmu import extract_mmmu_answer, mmmu_answers_equal, normalize_mmmu_extraction
from benchmarks.mathvista import extract_mathvista_answer, normalize_mathvista_extraction, safe_equal
from benchmarks.realworldqa import extract_realworldqa_answer, normalize_realworldqa_extraction
from benchmarks.scienceqa import extract_scienceqa_answer, normalize_scienceqa_extraction

try:
    from math_verify import parse, verify
except ImportError:
    parse = None
    verify = None


def _load_math_dependencies():
    from .grader import math_equal_process
    from .math_equivalent_MATH import is_equiv
    from .parse_utils_qwen import extract_answer as extract_fn

    return math_equal_process, is_equiv, extract_fn


def extract_true_answer(text, name="gsm8k"):
    if "gsm8k" in name:
        label = text.split("#### ")[1]
        return label
    if "MATH-500" in name:
        return text
    if "AIME_2024" in name:
        return text
    if "MathVista" in name:
        if text is None:
            return None
        if isinstance(text, str) and text.strip().lower() in {"", "none", "null", "nan", "n/a"}:
            return None
        return text
    if "LogicVista" in name:
        return None if text is None else str(text).strip()
    if "RealWorldQA" in name:
        return None if text is None else str(text).strip()
    if "MMStar" in name:
        return None if text is None else str(text).strip()
    if "MMMU" in name:
        return None if text is None else str(text).strip()
    if "HallusionBench" in name:
        return hallusionbench_label_to_text(text)
    if "ScienceQA-IMG" in name or "ScienceQA" in name:
        return None if text is None else str(text).strip()
    raise ValueError(f"Unknown dataset name: {name}")


def normalize_prediction(extraction, data_name="gsm8k", problem=None):
    if "MathVista" in data_name:
        if problem is None:
            raise ValueError("problem metadata is required for MathVista normalization")
        return normalize_mathvista_extraction(
            extraction=extraction,
            choices=problem["choices"],
            question_type=problem["question_type"],
            answer_type=problem["answer_type"],
            precision=problem["precision"],
        )
    if "MMStar" in data_name:
        if problem is None:
            raise ValueError("problem metadata is required for MMStar normalization")
        return normalize_mmstar_extraction(
            extraction=extraction,
            choices=problem["choices"],
        )
    if "MMMU" in data_name:
        if problem is None:
            raise ValueError("problem metadata is required for MMMU normalization")
        return normalize_mmmu_extraction(extraction, problem=problem)
    if "LogicVista" in data_name:
        return normalize_logicvista_extraction(extraction, problem=problem)
    if "RealWorldQA" in data_name:
        if problem is None:
            raise ValueError("problem metadata is required for RealWorldQA normalization")
        return normalize_realworldqa_extraction(extraction, problem=problem)
    if "HallusionBench" in data_name:
        return normalize_hallusionbench_extraction(extraction)
    if "ScienceQA-IMG" in data_name or "ScienceQA" in data_name:
        if problem is None:
            raise ValueError("problem metadata is required for ScienceQA normalization")
        return normalize_scienceqa_extraction(extraction, problem.get("choices") or [])
    return extraction


def judge_answer(
    input,
    label,
    data_name="gsm8k",
    extract=True,
    prompt_idx=0,
    model_name="Qwen2.5-7B-Instruct",
    problem=None,
):
    if "gsm8k" in data_name:
        if extract:
            input = extract_answer(input, data_name="gsm8k", prompt_idx=prompt_idx, model_name=model_name)
        return input == label

    if "MATH-500" in data_name:
        math_equal_process, is_equiv, _ = _load_math_dependencies()
        if extract:
            input = extract_answer(input, data_name="MATH-500", prompt_idx=prompt_idx, model_name=model_name)

        if parse is not None and verify is not None:
            hf_input = parse(input)
            if verify(label, hf_input):
                return True
        if math_equal_process((label, input)):
            return True
        if str(input) == str(label):
            return True
        if is_equiv(str(label), str(input)):
            return True
        return False

    if "AIME_2024" in data_name:
        if extract:
            input = extract_answer(input, data_name="AIME_2024", prompt_idx=prompt_idx, model_name=model_name)
            input = str(input)
            label = str(label)
        return input == label

    if "MathVista" in data_name:
        if problem is None:
            raise ValueError("problem metadata is required for MathVista scoring")
        if extract:
            input = extract_answer(
                input,
                data_name="MathVista",
                prompt_idx=prompt_idx,
                model_name=model_name,
                problem=problem,
            )
        prediction = normalize_prediction(input, data_name="MathVista", problem=problem)
        return safe_equal(prediction, label)

    if "MMStar" in data_name:
        if problem is None:
            raise ValueError("problem metadata is required for MMStar scoring")
        if extract:
            input = extract_answer(
                input,
                data_name="MMStar",
                prompt_idx=prompt_idx,
                model_name=model_name,
                problem=problem,
            )
        prediction = normalize_prediction(input, data_name="MMStar", problem=problem)
        return safe_equal(prediction, label)

    if "MMMU" in data_name:
        if problem is None:
            raise ValueError("problem metadata is required for MMMU scoring")
        if extract:
            input = extract_answer(
                input,
                data_name="MMMU",
                prompt_idx=prompt_idx,
                model_name=model_name,
                problem=problem,
            )
        prediction = normalize_prediction(input, data_name="MMMU", problem=problem)
        label = normalize_mmmu_extraction(label, problem=problem)
        return mmmu_answers_equal(prediction, label, problem=problem)

    if "LogicVista" in data_name:
        if extract:
            input = extract_answer(
                input,
                data_name="LogicVista",
                prompt_idx=prompt_idx,
                model_name=model_name,
                problem=problem,
            )
        prediction = normalize_prediction(input, data_name="LogicVista", problem=problem)
        label = normalize_logicvista_extraction(label, problem=problem)
        return safe_equal(prediction, label)

    if "RealWorldQA" in data_name:
        if problem is None:
            raise ValueError("problem metadata is required for RealWorldQA scoring")
        if extract:
            input = extract_answer(
                input,
                data_name="RealWorldQA",
                prompt_idx=prompt_idx,
                model_name=model_name,
                problem=problem,
            )
        prediction = normalize_prediction(input, data_name="RealWorldQA", problem=problem)
        label = normalize_realworldqa_extraction(label, problem=problem)
        return safe_equal(prediction, label)

    if "HallusionBench" in data_name:
        if extract:
            input = extract_answer(
                input,
                data_name="HallusionBench",
                prompt_idx=prompt_idx,
                model_name=model_name,
                problem=problem,
            )
        prediction = normalize_prediction(input, data_name="HallusionBench", problem=problem)
        label = hallusionbench_label_to_text(label)
        return safe_equal(prediction, label)

    if "ScienceQA-IMG" in data_name or "ScienceQA" in data_name:
        if problem is None:
            raise ValueError("problem metadata is required for ScienceQA scoring")
        if extract:
            input = extract_answer(
                input,
                data_name="ScienceQA",
                prompt_idx=prompt_idx,
                model_name=model_name,
                problem=problem,
            )
        prediction = normalize_prediction(input, data_name="ScienceQA", problem=problem)
        label = normalize_scienceqa_extraction(label, problem.get("choices") or [])
        return safe_equal(prediction, label)

    raise ValueError(f"Unknown dataset name: {data_name} for judge answer")


def extract_answer(text, data_name="gsm8k", prompt_idx=0, model_name="Qwen2.5-7B-Instruct", problem=None):
    if "gsm8k" in data_name:
        if prompt_idx == 0:
            if "qwen2.5-1.5b-instruct" in model_name.lower():
                return _extract_qwen25_1_5B_answer(text)
            return _extract_answer(text)
        if prompt_idx == 1:
            try:
                answer = json.loads(text.strip("` \n"))
                final_answer = answer.get("final answer", "")
                if not isinstance(final_answer, str):
                    final_answer = str(final_answer)
                return _extract_answer(final_answer)
            except json.JSONDecodeError:
                pattern = r'(?:final answer|my answer)"?:?\s*(.*?)[}<]'
                match = re.search(pattern, text, flags=re.I | re.M | re.DOTALL)
                if match:
                    return _extract_answer(match.group(1))
                return _extract_answer(text)
        raise ValueError(f"Unknown prompt index: {prompt_idx} for extract answer")

    if "MATH-500" in data_name:
        _, _, extract_fn = _load_math_dependencies()
        if prompt_idx == 0:
            return extract_fn(text, data_name="math")
        if prompt_idx == 1:
            try:
                answer = json.loads(text.strip("` \n"))
                final_answer = answer.get("final answer", "")
                if not isinstance(final_answer, str):
                    final_answer = str(final_answer)
                final_answer = final_answer.replace("\n", "").replace("\"", "").replace("\'", "")
                return final_answer
            except json.JSONDecodeError:
                text = text.replace("\n", "")
                pattern = r'(?:final answer|my answer)"?:?\s*(.*?)(}<|<\|)'
                match = re.search(pattern, text, flags=re.I | re.M | re.DOTALL)
                if match:
                    temp = match.group(1)
                    return temp.replace("\n", "").replace("\"", "").replace("\'", "")
                return None

    if "AIME_2024" in data_name:
        if prompt_idx == 0:
            return _extract_answer(text)
        if prompt_idx == 1:
            try:
                answer = json.loads(text.strip("` \n"))
                final_answer = answer.get("final answer", "")
                if not isinstance(final_answer, str):
                    final_answer = str(final_answer)
                return _extract_answer(final_answer)
            except json.JSONDecodeError:
                pattern = r'(?:final answer|my answer)"?:?\s*(.*?)[}<]'
                match = re.search(pattern, text, flags=re.I | re.M | re.DOTALL)
                if match:
                    return _extract_answer(match.group(1))
                return _extract_answer(text)

    if "MathVista" in data_name:
        return extract_mathvista_answer(text, prompt_idx=prompt_idx)

    if "LogicVista" in data_name:
        return extract_logicvista_answer(text, prompt_idx=prompt_idx)

    if "RealWorldQA" in data_name:
        return extract_realworldqa_answer(text, prompt_idx=prompt_idx)

    if "MMStar" in data_name:
        return extract_mmstar_answer(text, prompt_idx=prompt_idx)

    if "MMMU" in data_name:
        return extract_mmmu_answer(text, prompt_idx=prompt_idx)

    if "HallusionBench" in data_name:
        return extract_hallusionbench_answer(text, prompt_idx=prompt_idx)

    if "ScienceQA-IMG" in data_name or "ScienceQA" in data_name:
        return extract_scienceqa_answer(text, prompt_idx=prompt_idx)

    raise ValueError(f"Unknown dataset name: {data_name} for extract answer")


def extract_MATH_solution(solution_str: str):
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        processed_str = solution_str

    answer_pattern = r"<answer>.*?(\\boxed{.*}).*?</answer>"
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    if not matches:
        answer_pattern = r"\\boxed{(.*)}"
        matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    if not matches:
        print("[Error] No valid answer tags found")
        return None
    return matches[-1].group(1).strip()


def _extract_answer(text):
    if text is None:
        return None
    text = text.strip()

    def clean_number(num_str):
        num_str = re.sub(r"[$€£¥]", "", num_str)
        num_str = re.sub(r",", "", num_str)
        num_str = re.sub(r"\s", "", num_str)
        return num_str

    boxed_matches = list(re.finditer(r"\\boxed{(.*?)}", text, re.DOTALL))
    if boxed_matches:
        return clean_number(boxed_matches[-1].group(1))

    patterns = [
        r"(?i)(?:final answer|answer)\s*[:：]\s*([-+]?\d*\.?\d+)",
        r"[-+]?\d*\.?\d+",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return clean_number(matches[-1])
    return None


def _extract_qwen25_1_5B_answer(text):
    return _extract_answer(text)
