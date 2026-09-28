import re

import torch

try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs):
        return text

from methods.latentgrad.generation_utils import get_stop_token_ids, reset_multimodal_generation_state
from methods.latentgrad.optimize import optimized_generation
from prompts.vera_prompts import VERA_ANSWER_SYMBOL, get_vera_prompt


ORIGINAL_LATENTSEEK_TEXT_RULES = (
    "calculation_check",
    "answer_correct",
    "answer_completeness",
    "understanding_check",
)
ORIGINAL_LATENTSEEK_MULTIMODAL_RULES = ("visual_grounding_check",)


def _first_image(media):
    if media is None:
        return None
    image = media.get("image")
    if image is not None:
        return image
    images = media.get("images") or []
    return images[0] if images else None


def _build_visual_grounding_prompt(question, solution):
    verdict_rule = (
        f"Return only one line: '{VERA_ANSWER_SYMBOL}True' or "
        f"'{VERA_ANSWER_SYMBOL}False'. Do not add any explanation."
    )
    return f"""You are a critical verifier for visual question answering.

QUESTION:
{question}

PROPOSED SOLUTION:
{solution}

INSTRUCTIONS:
1. Inspect the image and identify the visual evidence required by the QUESTION.
2. Check whether the PROPOSED SOLUTION's final answer is supported by the image.
3. Return False if the answer contradicts the image, ignores visible evidence, relies on text-only guessing, or uses an object, count, spatial relation, chart value, or scene fact that is not supported by the image.
4. Return True only if the final answer is visually grounded and consistent with the image.
{verdict_rule}"""


class OriginalLatentSeekRewardModel:
    """LatentSeek verifier reward adapted to VLM benchmarks.

    The text mode keeps the upstream LatentSeek reward rule set unchanged. The
    multimodal mode adds exactly one visual-grounding verifier while leaving the
    four original text verifiers text-only.
    """

    def __init__(
        self,
        model,
        tokenizer,
        processor=None,
        data_name="gsm8k",
        device="cuda",
        rule_format_string=None,
        rule_mode="text",
        verifier_max_new_tokens=4096,
        use_compact_verifiers=False,
    ):
        if rule_mode not in {"text", "multimodal"}:
            raise ValueError(f"Unknown original_latentseek rule_mode: {rule_mode}")
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_name = data_name
        self.device = device
        self.rule_format_string = rule_format_string
        self.rule_mode = rule_mode
        self.verifier_max_new_tokens = int(verifier_max_new_tokens)
        self.use_compact_verifiers = bool(use_compact_verifiers)
        self.last_verifier_names = []

    def load_domain_specific_verifiers(self, media=None):
        veras = list(ORIGINAL_LATENTSEEK_TEXT_RULES)
        if self.rule_mode == "multimodal" and _first_image(media) is not None and self.processor is not None:
            veras.extend(ORIGINAL_LATENTSEEK_MULTIMODAL_RULES)
        self.last_verifier_names = list(veras)
        return veras

    def _build_prompt(self, verifier_name, question, solution):
        if verifier_name in ORIGINAL_LATENTSEEK_TEXT_RULES:
            return get_vera_prompt(
                verifier_name,
                question,
                solution,
                compact=self.use_compact_verifiers,
            )
        if verifier_name == "visual_grounding_check":
            return _build_visual_grounding_prompt(question, solution)
        raise KeyError(f"Unknown original_latentseek verifier: {verifier_name}")

    def _tokenize_text_prompt(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer([formatted], return_tensors="pt")

    def _tokenize_visual_prompt(self, prompt, image):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        formatted = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.processor(
            images=[image],
            text=[formatted],
            padding=True,
            return_tensors="pt",
        )

    def _run_single_verifier(self, prompt, image=None):
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.processor is not None and getattr(self.processor, "tokenizer", None) is not None:
            if self.processor.tokenizer.pad_token_id is None:
                self.processor.tokenizer.pad_token_id = self.tokenizer.pad_token_id

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            if image is not None and self.processor is not None:
                batch_inputs = self._tokenize_visual_prompt(prompt, image)
            else:
                batch_inputs = self._tokenize_text_prompt(prompt)

            batch_inputs = {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in batch_inputs.items()
            }
            prompt_length = batch_inputs["input_ids"].shape[1]

            reset_multimodal_generation_state(self.model)
            with torch.no_grad():
                outputs = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=self.verifier_max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=get_stop_token_ids(self.tokenizer),
                )
        finally:
            self.tokenizer.padding_side = original_padding_side

        if hasattr(outputs, "sequences"):
            outputs = outputs.sequences
        response_tokens = outputs[0][prompt_length:] if outputs.shape[1] > prompt_length else outputs[0]
        return self.tokenizer.decode(
            response_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def get_verifications(self, question, solution, media=None, problem=None):
        del problem
        image = _first_image(media)
        verifications = {}
        for verifier_name in self.load_domain_specific_verifiers(media=media):
            prompt = self._build_prompt(verifier_name, question, solution)
            verifier_image = image if verifier_name in ORIGINAL_LATENTSEEK_MULTIMODAL_RULES else None
            response = self._run_single_verifier(prompt, image=verifier_image)
            verifications[verifier_name] = self.extract_verifier_approval(response)
        return verifications

    def extract_verifier_approval(self, verifier_response):
        vera_answer_symbol = VERA_ANSWER_SYMBOL.lower()
        pattern = re.compile(
            r".*{}(.*)".format(re.escape(vera_answer_symbol)),
            flags=re.DOTALL | re.IGNORECASE,
        )
        match = pattern.search(verifier_response)
        answer = match.group(1).strip() if match else None

        if not answer:
            tail_matches = re.findall(r"\b(true|false)\b", verifier_response[-160:], flags=re.IGNORECASE)
            if tail_matches:
                return tail_matches[-1].lower() == "true"
            all_matches = re.findall(r"\b(true|false)\b", verifier_response, flags=re.IGNORECASE)
            if all_matches and len({item.lower() for item in all_matches}) == 1:
                return all_matches[-1].lower() == "true"
            print(
                colored(
                    f"WARNING in extract_verifier_approval: {answer=} with {type(answer)=}, "
                    f"and full verifier_response (length {len(verifier_response)}): "
                    f"\n{'-' * 30}\n{verifier_response}\n{'-' * 30}",
                    "yellow",
                )
            )
            return False

        answer = answer.replace("*", "").strip().lower()
        if "true" in answer:
            return True
        if "false" in answer:
            return False
        answer_parts = answer.split()
        if not answer_parts:
            return False
        first_word = answer_parts[0]
        if "true" in first_word:
            return True
        if "false" in first_word:
            return False
        print(
            colored(
                f"WARNING in extract_verifier_approval: {answer=} is not true/false. "
                f"Full verifier_response:\n{'-' * 30}\n{verifier_response}\n{'-' * 30}",
                "yellow",
            )
        )
        return False

    def get_reward(self, question, solution, media=None, problem=None):
        verifications = self.get_verifications(question, solution, media=media, problem=problem)
        reward = 0.0
        reward_list = self.get_reward_list()
        total = 0.0
        for verifier_name, verifier_approval in verifications.items():
            weight = reward_list[verifier_name]
            total += weight
            if verifier_approval:
                print(colored(f"Verifier {verifier_name} approved the solution.", "green"))
            else:
                print(colored(f"Verifier {verifier_name} disapproved the solution.", "red"))
                reward -= weight

        if self.rule_format_string is not None:
            format_approval = self.get_rule_format_verify(solution)
            if format_approval:
                print(colored("Verifier Rule Format approved the solution.", "green"))
            else:
                print(colored("Verifier Rule Format disapproved the solution.", "red"))
                reward -= 2.0

        return reward / max(total, 1.0)

    def get_rule_format_verify(self, solution):
        return len(list(re.finditer(self.rule_format_string, solution, re.DOTALL))) > 0

    def get_reward_list(self):
        return {
            "calculation_check": 2,
            "answer_correct": 1,
            "answer_completeness": 2,
            "understanding_check": 1,
            "visual_grounding_check": 2,
        }


def original_latentseek_generation(
    reward_model,
    model,
    tokenizer,
    device,
    question,
    input_text,
    original_answer,
    original_hidden_states_list,
    input_ids,
    start_index=0,
    prompt_inputs=None,
    media=None,
    problem=None,
    max_num_steps=10,
    max_new_tokens=1024,
    lr=0.03,
    grad_clip=None,
    k=0.1,
    reward_threshold=-0.2,
    latent_start_strategy="full",
):
    update_position_strategy = "head"
    if latent_start_strategy == "tail":
        update_position_strategy = "tail"

    optimized_output, reward_history, original_length, optimized_length, update_length = optimized_generation(
        reward_model=reward_model,
        model=model,
        tokenizer=tokenizer,
        device=device,
        question=question,
        input_text=input_text,
        original_answer=original_answer,
        original_hidden_states_list=original_hidden_states_list,
        input_ids=input_ids,
        start_index=start_index,
        prompt_inputs=prompt_inputs,
        media=media,
        problem=problem,
        max_num_steps=max_num_steps,
        max_new_tokens=max_new_tokens,
        lr=lr,
        grad_clip=grad_clip,
        k=k,
        reward_threshold=reward_threshold,
        latent_reg_lambda=0.0,
        update_position_strategy=update_position_strategy,
    )

    baseline_info = {
        "mode": "original_latentseek",
        "rule_mode": getattr(reward_model, "rule_mode", None),
        "verifier_names": list(getattr(reward_model, "last_verifier_names", []) or []),
        "max_num_steps": int(max_num_steps),
        "reward_threshold": float(reward_threshold),
        "verifier_max_new_tokens": int(getattr(reward_model, "verifier_max_new_tokens", 0)),
        "use_compact_verifiers": bool(getattr(reward_model, "use_compact_verifiers", False)),
        "reward_model": "original_latentseek_rule_verifier_reward",
    }
    return optimized_output, reward_history, original_length, optimized_length, update_length, baseline_info
