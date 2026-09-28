"""
Visual Likelihood Reward for multimodal test-time optimization.
"""
import json
import math
import re

import torch
import torch.nn.functional as F

from methods.latentgrad.generation_utils import GENERATION_MEDIA_KEYS


def _extract_short_answer_for_scoring(text, problem=None):
    if text is None:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            for key in ["final answer", "final_answer", "answer"]:
                if key in obj:
                    val = obj[key]
                    return str(val).strip() if val is not None else None
        except Exception:
            pass

    for pattern in [
        r'"final answer"\s*:\s*"?(.*?)"?\s*[,}\n]',
        r'"final_answer"\s*:\s*"?(.*?)"?\s*[,}\n]',
        r"(?:final answer)\s*[:：]\s*(.+)",
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip().strip('"').strip("'")

    fallback = text.strip().strip('"').strip("'")
    return fallback if fallback else None


def _normalize_to_choice_letter(answer, choices):
    if answer is None or not choices:
        return None

    letters = [chr(ord("A") + i) for i in range(len(choices))]
    clean = answer.strip().upper().strip("()").strip()
    if clean in letters:
        return clean

    for idx, choice in enumerate(choices):
        if str(answer).strip().lower() == str(choice).strip().lower():
            return letters[idx]

    return clean if len(clean) == 1 and clean.isalpha() else None


def _normalize_short_answer_for_dataset(short, problem):
    if short is None or problem is None:
        return None

    dataset_name = problem.get("dataset_name")
    question_type = problem.get("question_type")
    answer_type = problem.get("answer_type")

    if dataset_name == "MathVista":
        from benchmarks.mathvista import normalize_mathvista_extraction

        normalized = normalize_mathvista_extraction(
            extraction=short,
            choices=problem.get("choices") or [],
            question_type=question_type,
            answer_type=answer_type,
            precision=problem.get("precision"),
            ignore_empty_extractions=True,
        )
        return normalized

    if dataset_name == "HallusionBench":
        from benchmarks.hallusionbench import normalize_hallusionbench_extraction

        return normalize_hallusionbench_extraction(short, ignore_empty_extractions=True)

    if dataset_name == "ScienceQA":
        from benchmarks.scienceqa import normalize_scienceqa_extraction

        return normalize_scienceqa_extraction(short, problem.get("choices") or [], ignore_empty_extractions=True)

    if dataset_name == "MMMU":
        from benchmarks.mmmu import normalize_mmmu_extraction

        return normalize_mmmu_extraction(short, problem=problem, ignore_empty_extractions=True)

    if question_type == "multi_choice" and problem.get("choices"):
        letter = _normalize_to_choice_letter(short, problem.get("choices") or [])
        return letter

    if answer_type == "integer":
        try:
            return str(int(float(str(short).replace(",", "").strip())))
        except Exception:
            return None

    if answer_type == "float":
        try:
            return str(float(str(short).replace(",", "").strip()))
        except Exception:
            return None

    return str(short).strip()


def extract_normalized_short_answer_for_scoring(text, problem):
    short = _extract_short_answer_for_scoring(text, problem)
    if short is None:
        return None
    normalized_short = _normalize_short_answer_for_dataset(short, problem)
    if normalized_short is None or not str(normalized_short).strip():
        return None
    return str(normalized_short).strip()


class VisualLikelihoodReward:
    def __init__(
        self,
        model,
        processor,
        tokenizer,
        device="cuda",
        reward_mode="likelihood",
        prior_mode="text_only",
        prior_weight=1.0,
        contrastive_temperature=1.0,
    ):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        self.reward_mode = reward_mode
        self.prior_mode = prior_mode
        self.prior_weight = prior_weight
        self.contrastive_temperature = max(float(contrastive_temperature), 1e-6)

    def _get_output_head(self):
        if hasattr(self.model, "get_output_embeddings"):
            output_head = self.model.get_output_embeddings()
            if output_head is not None:
                return output_head
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        language_model = getattr(self.model, "language_model", None)
        if language_model is not None:
            if hasattr(language_model, "get_output_embeddings"):
                output_head = language_model.get_output_embeddings()
                if output_head is not None:
                    return output_head
            if hasattr(language_model, "lm_head"):
                return language_model.lm_head
        return None

    def _extract_logits(self, outputs):
        logits = getattr(outputs, "logits", None)
        if logits is not None:
            return logits

        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states:
                hidden = hidden_states[-1]
        if hidden is None and isinstance(outputs, (tuple, list)) and outputs:
            first = outputs[0]
            if torch.is_tensor(first) and first.ndim == 3:
                hidden = first
        if hidden is None:
            raise AttributeError(
                "Model output does not contain logits or last_hidden_state for completion scoring."
            )

        output_head = self._get_output_head()
        if output_head is None:
            raise AttributeError(
                "Model output does not contain logits and no output embedding/lm_head is available."
            )
        return output_head(hidden)

    def _resolve_media(self, media):
        images = media.get("images") if media else None
        image = media.get("image") if media else None
        if image is None and images:
            image = images[0]
        return images, image

    def _build_answer_only_prompt(self, problem, media, include_image=True):
        dataset_name = problem.get("dataset_name") if problem else None
        if dataset_name == "MMStar":
            from benchmarks.mmstar import build_mmstar_answer_only_text

            prompt_text = build_mmstar_answer_only_text(problem)
        elif dataset_name == "MMMU":
            from benchmarks.mmmu import build_mmmu_answer_only_text

            prompt_text = build_mmmu_answer_only_text(problem)
        elif dataset_name == "LogicVista":
            from benchmarks.logicvista import build_logicvista_answer_only_text

            prompt_text = build_logicvista_answer_only_text(problem)
        elif dataset_name == "RealWorldQA":
            from benchmarks.realworldqa import build_realworldqa_answer_only_text

            prompt_text = build_realworldqa_answer_only_text(problem)
        elif dataset_name == "HallusionBench":
            from benchmarks.hallusionbench import build_hallusionbench_answer_only_text

            prompt_text = build_hallusionbench_answer_only_text(problem)
        elif dataset_name == "ScienceQA":
            from benchmarks.scienceqa import build_scienceqa_answer_only_text

            prompt_text = build_scienceqa_answer_only_text(problem)
        else:
            from benchmarks.mathvista import build_mathvista_answer_only_text

            prompt_text = build_mathvista_answer_only_text(problem)

        image = None
        if media:
            image = media.get("image")
            if image is None:
                images = media.get("images")
                if images:
                    image = images[0]
        user_content = []
        if include_image and image is not None:
            user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": prompt_text})
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise visual math solver. Analyze the image carefully and "
                    "solve the problem."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        return messages

    def _compute_completion_logprob(self, prompt_messages, target_text, media=None, include_image=True):
        if self.processor is None:
            return 0.0

        images, image = self._resolve_media(media)
        if include_image and image is None and not images:
            return 0.0

        prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        if include_image:
            prompt_batch = self.processor(images=images or [image], text=[prompt_text], return_tensors="pt")
        else:
            prompt_batch = self.processor(text=[prompt_text], return_tensors="pt")
        prompt_batch = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in prompt_batch.items()
        }

        cand_ids = self.tokenizer(target_text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        if cand_ids.shape[1] == 0:
            return -10.0

        input_ids = torch.cat([prompt_batch["input_ids"], cand_ids], dim=1)
        attention_mask = torch.cat([prompt_batch["attention_mask"], torch.ones_like(cand_ids)], dim=1)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        for key in GENERATION_MEDIA_KEYS:
            if key in prompt_batch:
                model_inputs[key] = prompt_batch[key]

        with torch.no_grad():
            model_type = getattr(getattr(self.model, "config", None), "model_type", None)
            if (
                not include_image
                and "pixel_values" not in model_inputs
                and hasattr(self.model, "language_model")
                and model_type == "internvl_chat"
            ):
                outputs = self.model.language_model(**model_inputs)
            else:
                outputs = self.model(**model_inputs)
            prompt_len = prompt_batch["input_ids"].shape[1]
            logits = self._extract_logits(outputs)[:, prompt_len - 1 : -1, :]
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            token_logprobs = logprobs.gather(-1, cand_ids.to(logprobs.device).unsqueeze(-1)).squeeze(-1)

        return token_logprobs.mean().item()

    def score_completion_logprob(self, prompt_messages, target_text, media=None, include_image=True):
        return self._compute_completion_logprob(
            prompt_messages=prompt_messages,
            target_text=target_text,
            media=media,
            include_image=include_image,
        )

    def _compute_answer_logprob(self, media, problem, answer_text, include_image=True):
        messages = self._build_answer_only_prompt(problem, media, include_image=include_image)
        return self._compute_completion_logprob(
            prompt_messages=messages,
            target_text=answer_text,
            media=media,
            include_image=include_image,
        )

    def score_short_answer(self, answer_text, media, problem):
        if answer_text is None or not str(answer_text).strip():
            return -10.0, None, None

        short = extract_normalized_short_answer_for_scoring(str(answer_text), problem)
        if short is None:
            return -10.0, None, None

        visual_logprob = self._compute_answer_logprob(media, problem, short, include_image=True)
        prior_logprob = None
        if self.reward_mode == "contrastive":
            include_image = self.prior_mode != "text_only"
            prior_logprob = self._compute_answer_logprob(media, problem, short, include_image=include_image)
            score = visual_logprob - self.prior_weight * prior_logprob
        else:
            score = visual_logprob

        return score, visual_logprob, prior_logprob

    def score_answer(self, answer_text, media, problem):
        score, _, _ = self.score_short_answer(answer_text, media, problem)
        return score

    def get_reward(self, question, solution, media=None, problem=None, **kwargs):
        if problem is None:
            return 0.0
        if media is None:
            media = {}
        if problem.get("dataset_name") == "HallusionBench" and str(problem.get("visual_input")) == "0":
            pass
        elif media.get("image") is None and not media.get("images"):
            return 0.0

        score = self.score_answer(solution, media, problem)
        if self.reward_mode == "contrastive":
            reward = math.tanh(score / self.contrastive_temperature)
        else:
            reward = max(-1.0, min(1.0, (score + 2.0) / 2.0))
        return reward
