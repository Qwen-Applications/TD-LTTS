"""
Confidence-based self-reward for test-time latent optimization.
"""
import json
import re

import torch
import torch.nn.functional as F


def _extract_json_answer(text):
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
                    return str(obj[key]).strip()
        except Exception:
            pass
    patterns = [
        r'"final answer"\s*:\s*"?(.*?)"?\s*[,}\n]',
        r'"final_answer"\s*:\s*"?(.*?)"?\s*[,}\n]',
        r"(?:final answer)\s*[:：]\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip().strip('"').strip("'")
    return None


class ConfidenceReward:
    def __init__(
        self,
        model,
        tokenizer,
        processor=None,
        device="cuda",
        data_name="MathVista",
        entropy_weight=0.4,
        logprob_weight=0.3,
        format_weight=0.2,
        visual_weight=0.1,
        entropy_threshold=1.5,
        logprob_normalizer=-2.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device
        self.data_name = data_name

        self.entropy_weight = entropy_weight
        self.logprob_weight = logprob_weight
        self.format_weight = format_weight
        self.visual_weight = visual_weight
        self.entropy_threshold = entropy_threshold
        self.logprob_normalizer = logprob_normalizer

    def _compute_token_entropy(self, logits):
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy.mean().item()

    def _compute_sequence_logprob(self, logits, token_ids):
        log_probs = F.log_softmax(logits.float(), dim=-1)
        selected = log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        return selected.mean().item()

    def _check_format_compliance(self, text, problem=None):
        if text is None or not text.strip():
            return -1.0

        answer = _extract_json_answer(text)
        if answer is None or not answer.strip():
            return -0.5

        if problem is not None:
            question_type = problem.get("question_type")
            answer_type = problem.get("answer_type")
            choices = problem.get("choices") or []

            if question_type == "multi_choice" and choices:
                letters = [chr(ord("A") + i) for i in range(len(choices))]
                clean = answer.strip().upper().strip("()").strip()
                if clean in letters or answer in choices:
                    return 1.0
                return -0.3
            if answer_type == "integer":
                try:
                    int(float(answer.replace(",", "").strip()))
                    return 1.0
                except Exception:
                    return -0.3
            if answer_type == "float":
                try:
                    float(answer.replace(",", "").strip())
                    return 1.0
                except Exception:
                    return -0.3

        return 0.5

    def _compute_visual_consistency(self, optimized_hidden, original_hidden):
        if optimized_hidden is None or original_hidden is None:
            return 0.0
        opt_flat = optimized_hidden.reshape(-1).float()
        orig_flat = original_hidden.reshape(-1).float()
        return F.cosine_similarity(opt_flat.unsqueeze(0), orig_flat.unsqueeze(0)).item()

    def get_reward_from_logits(self, logits, token_ids, text, problem=None, optimized_hidden=None, original_hidden=None):
        entropy = self._compute_token_entropy(logits)
        entropy_score = max(-1.0, min(1.0, 1.0 - entropy / self.entropy_threshold))

        logprob = self._compute_sequence_logprob(logits, token_ids)
        logprob_score = max(-1.0, min(1.0, -logprob / self.logprob_normalizer))

        format_score = self._check_format_compliance(text, problem=problem)
        visual_score = self._compute_visual_consistency(optimized_hidden, original_hidden)

        reward = (
            self.entropy_weight * entropy_score
            + self.logprob_weight * logprob_score
            + self.format_weight * format_score
            + self.visual_weight * visual_score
        )
        reward = max(-1.0, min(1.0, reward))
        return reward, {
            "entropy": entropy,
            "entropy_score": entropy_score,
            "logprob": logprob,
            "logprob_score": logprob_score,
            "format_score": format_score,
            "visual_score": visual_score,
            "composite_reward": reward,
        }

    def get_reward(
        self,
        question,
        solution,
        media=None,
        problem=None,
        logits=None,
        token_ids=None,
        optimized_hidden=None,
        original_hidden=None,
    ):
        if logits is not None and token_ids is not None:
            reward, _ = self.get_reward_from_logits(
                logits=logits,
                token_ids=token_ids,
                text=solution,
                problem=problem,
                optimized_hidden=optimized_hidden,
                original_hidden=original_hidden,
            )
            return reward

        return self._check_format_compliance(solution, problem=problem) * 0.5
