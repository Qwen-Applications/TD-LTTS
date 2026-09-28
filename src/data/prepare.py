"""Per-example preparation helpers."""

from benchmarks.hallusionbench import (
    build_hallusionbench_question_text,
    hallusionbench_label_to_text,
    is_hallusionbench_answer_available,
)
from benchmarks.logicvista import build_logicvista_question_text, is_logicvista_answer_available
from benchmarks.mmstar import build_mmstar_question_text, parse_mmstar_question
from benchmarks.mmmu import build_mmmu_question_text, parse_mmmu_options
from benchmarks.mathvista import build_mathvista_question_text, is_mathvista_answer_available
from benchmarks.realworldqa import (
    build_realworldqa_question_text,
    is_realworldqa_answer_available,
    parse_realworldqa_question,
)
from benchmarks.scienceqa import answer_index_to_letter, build_scienceqa_question_text
from prompts import (
    AIME_2024_prompt,
    HallusionBench_prompt,
    LogicVista_prompt,
    MATH_500_prompt,
    MMStar_prompt,
    MMMU_prompt,
    MathVista_prompt,
    RealWorldQA_prompt,
    ScienceQA_prompt,
    gsm8k_prompt,
)


def prepare_example(example, data_name_or_path, tokenizer, prompt_idx, processor=None):
    if "gsm8k" in data_name_or_path:
        question = example["question"]
        answer = example["answer"]
        messages = gsm8k_prompt(question, prompt_idx)
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {
            "formatted": formatted,
            "question": question,
            "answer": answer,
            "example_id": None,
            "model_inputs": None,
            "problem": None,
        }

    if "MATH-500" in data_name_or_path:
        question = example["problem"]
        answer = example["answer"]
        messages = MATH_500_prompt(question, prompt_idx)
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {
            "formatted": formatted,
            "question": question,
            "answer": answer,
            "example_id": None,
            "model_inputs": None,
            "problem": None,
        }

    if "AIME_2024" in data_name_or_path:
        question = example["Problem"]
        answer = example["Answer"]
        messages = AIME_2024_prompt(question, prompt_idx)
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {
            "formatted": formatted,
            "question": question,
            "answer": answer,
            "example_id": None,
            "model_inputs": None,
            "problem": None,
        }

    if "MathVista" in data_name_or_path:
        if processor is None:
            raise ValueError("processor is required for MathVista")

        image = example.get("decoded_image") or example.get("image")
        if image is None:
            raise ValueError("MathVista example does not contain a decoded image")
        messages = MathVista_prompt(example, image=image, prompt_idx=prompt_idx)
        formatted = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = processor(
            images=[image],
            text=[formatted],
            return_tensors="pt",
        )
        problem = {
            "pid": example["pid"],
            "question": example["question"],
            "choices": example["choices"],
            "unit": example["unit"],
            "precision": example["precision"],
            "answer": example["answer"],
            "question_type": example["question_type"],
            "answer_type": example["answer_type"],
            "metadata": example["metadata"],
            "query": example["query"],
        }
        answer = example["answer"] if is_mathvista_answer_available(example["answer"]) else None
        return {
            "formatted": formatted,
            "question": build_mathvista_question_text(problem),
            "answer": answer,
            "example_id": example["pid"],
            "model_inputs": model_inputs,
            "problem": problem,
            "image": image,
        }

    if "MMStar" in data_name_or_path:
        if processor is None:
            raise ValueError("processor is required for MMStar")

        image = example.get("decoded_image") or example.get("image")
        if image is None:
            raise ValueError("MMStar example does not contain a decoded image")

        question, choices, hint = parse_mmstar_question(example["question"])
        problem = {
            "dataset_name": "MMStar",
            "index": example["index"],
            "question": question,
            "hint": hint,
            "choices": choices,
            "question_type": "multi_choice",
            "answer_type": "text",
            "precision": None,
            "answer": example["answer"],
            "category": example.get("category"),
            "l2_category": example.get("l2_category"),
            "meta_info": example.get("meta_info"),
            "raw_question": example["question"],
        }
        messages = MMStar_prompt(problem, image=image, prompt_idx=prompt_idx)
        formatted = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = processor(
            images=[image],
            text=[formatted],
            return_tensors="pt",
        )
        return {
            "formatted": formatted,
            "question": build_mmstar_question_text(problem),
            "answer": example["answer"],
            "example_id": str(example["index"]),
            "model_inputs": model_inputs,
            "problem": problem,
            "image": image,
            "images": [image],
        }

    if "MMMU" in data_name_or_path:
        if processor is None:
            raise ValueError("processor is required for MMMU")

        images = [example.get(f"image_{idx}") for idx in range(1, 8) if example.get(f"image_{idx}") is not None]
        if len(images) != 1:
            raise ValueError(f"MMMU example must contain exactly one image after filtering, got {len(images)}")
        image = images[0]
        source_question_type = str(example.get("question_type") or "").strip()
        question_type = "multi_choice" if source_question_type == "multiple-choice" else "open"
        choices = parse_mmmu_options(example.get("options"))
        problem = {
            "dataset_name": "MMMU",
            "id": example.get("id"),
            "question": example["question"],
            "choices": choices,
            "question_type": question_type,
            "answer_type": "text",
            "answer": example.get("answer"),
            "source_question_type": source_question_type,
            "subject": example.get("subject"),
            "subfield": example.get("subfield"),
            "topic_difficulty": example.get("topic_difficulty"),
            "img_type": example.get("img_type"),
            "explanation": example.get("explanation"),
        }
        messages = MMMU_prompt(problem, image=image, prompt_idx=prompt_idx)
        formatted = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = processor(
            images=[image],
            text=[formatted],
            return_tensors="pt",
        )
        return {
            "formatted": formatted,
            "question": build_mmmu_question_text(problem),
            "answer": example.get("answer"),
            "example_id": example.get("id"),
            "model_inputs": model_inputs,
            "problem": problem,
            "image": image,
            "images": [image],
        }

    if "LogicVista" in data_name_or_path:
        if processor is None:
            raise ValueError("processor is required for LogicVista")

        images = example.get("images") or []
        if not images:
            raise ValueError("LogicVista example does not contain images")
        image = images[0]
        problem = {
            "dataset_name": "LogicVista",
            "problem": example["problem"],
            "answer": example["answer"],
            "question_type": "multi_choice",
            "answer_type": "text",
        }
        answer = example["answer"] if is_logicvista_answer_available(problem) else None
        messages = LogicVista_prompt(problem, image=image, prompt_idx=prompt_idx)
        formatted = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = processor(
            images=[image],
            text=[formatted],
            return_tensors="pt",
        )
        return {
            "formatted": formatted,
            "question": build_logicvista_question_text(problem),
            "answer": answer,
            "example_id": None,
            "model_inputs": model_inputs,
            "problem": problem,
            "image": image,
            "images": [image],
        }

    if "RealWorldQA" in data_name_or_path:
        if processor is None:
            raise ValueError("processor is required for RealWorldQA")

        image = example.get("decoded_image") or example.get("image")
        if image is None:
            raise ValueError("RealWorldQA example does not contain a decoded image")

        question, choices, hint, question_type = parse_realworldqa_question(example["question"])
        problem = {
            "dataset_name": "RealWorldQA",
            "question": question,
            "hint": hint,
            "choices": choices,
            "question_type": question_type,
            "answer_type": "text",
            "answer": example["answer"],
            "image_path": example.get("image_path"),
            "raw_question": example["question"],
        }
        answer = example["answer"] if is_realworldqa_answer_available(example["answer"]) else None
        messages = RealWorldQA_prompt(problem, image=image, prompt_idx=prompt_idx)
        formatted = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = processor(
            images=[image],
            text=[formatted],
            return_tensors="pt",
        )
        return {
            "formatted": formatted,
            "question": build_realworldqa_question_text(problem),
            "answer": answer,
            "example_id": example.get("image_path") or None,
            "model_inputs": model_inputs,
            "problem": problem,
            "image": image,
            "images": [image],
        }

    if "HallusionBench" in data_name_or_path:
        if processor is None:
            raise ValueError("processor is required for HallusionBench")

        image = example.get("image")
        visual_input = str(example.get("visual_input", "")).strip()
        has_image = visual_input == "1" and image is not None
        problem = {
            "dataset_name": "HallusionBench",
            "question": example["question"],
            "question_type": "yes_no",
            "answer_type": "text",
            "answer": hallusionbench_label_to_text(example.get("gt_answer")),
            "category": example.get("category"),
            "l2_category": example.get("subcategory"),
            "visual_input": visual_input,
            "source_split": example.get("source_split"),
            "set_id": str(example.get("set_id")),
            "figure_id": str(example.get("figure_id")),
            "question_id": str(example.get("question_id")),
            "sample_note": example.get("sample_note"),
            "filename": example.get("filename"),
            "gt_answer_details": example.get("gt_answer_details"),
        }
        messages = HallusionBench_prompt(problem, image=image if has_image else None, prompt_idx=prompt_idx)
        formatted = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if has_image:
            model_inputs = processor(
                images=[image],
                text=[formatted],
                return_tensors="pt",
            )
            images = [image]
        else:
            model_inputs = processor(
                text=[formatted],
                return_tensors="pt",
            )
            images = []
        answer = problem["answer"] if is_hallusionbench_answer_available(problem["answer"]) else None
        example_id = (
            f"HallusionBench_{example.get('source_split')}_{example.get('category')}_"
            f"{example.get('set_id')}_{example.get('figure_id')}_{example.get('question_id')}"
        )
        return {
            "formatted": formatted,
            "question": build_hallusionbench_question_text(problem),
            "answer": answer,
            "example_id": example_id,
            "model_inputs": model_inputs,
            "problem": problem,
            "image": image if has_image else None,
            "images": images,
        }

    if "ScienceQA-IMG" in data_name_or_path or "ScienceQA" in data_name_or_path:
        if processor is None:
            raise ValueError("processor is required for ScienceQA-IMG")

        image = example.get("image")
        if image is None:
            raise ValueError("ScienceQA-IMG example does not contain an image")

        choices = example.get("choices") or []
        answer = answer_index_to_letter(example.get("answer"), choices)
        problem = {
            "dataset_name": "ScienceQA",
            "question": example["question"],
            "hint": example.get("hint"),
            "choices": choices,
            "question_type": "multi_choice",
            "answer_type": "text",
            "answer": answer,
            "task": example.get("task"),
            "grade": example.get("grade"),
            "subject": example.get("subject"),
            "topic": example.get("topic"),
            "category": example.get("category"),
            "skill": example.get("skill"),
            "lecture": example.get("lecture"),
            "solution": example.get("solution"),
        }
        messages = ScienceQA_prompt(problem, image=image, prompt_idx=prompt_idx)
        formatted = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = processor(
            images=[image],
            text=[formatted],
            return_tensors="pt",
        )
        example_id = f"ScienceQA_{example.get('subject')}_{example.get('topic')}_{example.get('grade')}_{answer}_{hash(example['question']) & 0xffffffff}"
        return {
            "formatted": formatted,
            "question": build_scienceqa_question_text(problem),
            "answer": answer,
            "example_id": example_id,
            "model_inputs": model_inputs,
            "problem": problem,
            "image": image,
            "images": [image],
        }

    raise ValueError(f"Unsupported dataset: {data_name_or_path}")
