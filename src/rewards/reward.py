import re
import torch

try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs):
        return text

from methods.latentgrad.generation_utils import get_stop_token_ids, reset_multimodal_generation_state
from benchmarks.mathvista import extract_mathvista_answer, normalize_mathvista_extraction
from prompts.vera_prompts import VERA_ANSWER_SYMBOL, get_vera_prompt

class RewardModel(object):
    def __init__(
            self,
            model,
            tokenizer,
            processor=None,
            data_name: str = "gsm8k",
            device: str = "cuda",
            rule_format_string: str = None,
            prompt_idx: int = 0,
            structured_reward_weight: float = 0.0,
            counterfactual_reward_weight: float = 0.0,
            verifier_max_new_tokens: int = 64,
            use_compact_verifiers: bool = True,
        ):
        """
        Args:
            model: the model to use for reward prediction
            tokenizer: the tokenizer to use for reward prediction
            data_name
            device
            rule_format_string: str, the answer format that the solution should follow
        """

        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor

        self.type = type

        self.data_name = data_name
        self.device = device
        self.rule_format_string = rule_format_string
        self.prompt_idx = prompt_idx
        self.structured_reward_weight = structured_reward_weight
        self.counterfactual_reward_weight = counterfactual_reward_weight
        self.verifier_max_new_tokens = verifier_max_new_tokens
        self.use_compact_verifiers = use_compact_verifiers


    def _solution_has_calculation(self, solution):
        if solution is None:
            return False
        patterns = [
            r"=",
            r"\\frac",
            r"\d\s*[\+\-\*/]\s*\d",
            r"\bsolve\b",
            r"\bcalculate\b",
            r"\bsubstitute\b",
            r"\btherefore\b",
            r"\bso\b",
        ]
        return any(re.search(pattern, solution, flags=re.IGNORECASE) for pattern in patterns)


    def load_domain_specific_verifiers(self, problem=None, solution=None):
        veras = [
            "answer_correct",
            "answer_completeness",
            "understanding_check",
        ]
        if self._solution_has_calculation(solution):
            veras.insert(0, "calculation_check")
        return veras


    def get_verifications(self, question: str, solution: str, media=None, problem=None):
        '''
        Get verifications from different verifiers (batched: all 4 verifiers in one generate call).

        Args:
            question: str, question
            solution: str, solution

        Returns:
            verifications: dict, verifier_name -> verifier_approval
        '''
        veras = self.load_domain_specific_verifiers(problem=problem, solution=solution)
        image = None
        if media is not None:
            image = media.get("image")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        texts = []
        images = []
        compact_prompt = self.use_compact_verifiers and ("MathVista" in self.data_name or image is not None)
        for vera_type in veras:
            vera_prompt = get_vera_prompt(vera_type, question, solution, compact=compact_prompt)
            if image is not None and self.processor is not None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": vera_prompt},
                        ],
                    }
                ]
                texts.append(
                    self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )
                images.append(image)
            else:
                messages = [{"role": "user", "content": vera_prompt}]
                texts.append(
                    self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )

        if image is not None and self.processor is not None:
            batch_inputs = self.processor(
                images=images,
                text=texts,
                padding=True,
                return_tensors="pt",
            )
        else:
            batch_inputs = self.tokenizer(
                texts,
                padding=True,
                return_tensors="pt",
            )

        batch_inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch_inputs.items()
        }

        reset_multimodal_generation_state(self.model)
        outputs = self.model.generate(
            **batch_inputs,
            max_new_tokens=self.verifier_max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=get_stop_token_ids(self.tokenizer),
        )

        self.tokenizer.padding_side = orig_padding_side

        verifications = dict()
        prompt_length = batch_inputs["input_ids"].shape[1]
        for i, vera_type in enumerate(veras):
            generated_tokens = outputs[i][prompt_length:]
            response = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            verifications[vera_type] = self.extract_verifier_approval(response)

        return verifications


    def extract_verifier_approval(self, verifier_response):
        '''
        Extract verifier approval from verifier response.

        Args:
            verifier_response: str, verifier response

        Returns:
            verifier_approval: bool, verifier approval
        '''
        vera_answer_symbol = VERA_ANSWER_SYMBOL.lower()
        pattern = re.compile(
            r'.*{}(.*)'.format(re.escape(vera_answer_symbol)),
            flags=re.DOTALL | re.IGNORECASE
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

        if not answer:
            print(colored(f"WARNING in extract_verifier_approval: {answer=} with {type(answer)=}, "
                        f"and full verifier_response (length {len(verifier_response)}): "
                        f"\n{'-' * 30}\n{verifier_response}\n{'-' * 30} (WARNING in extract_verifier_approval)\n", "yellow"))
            return False

        answer = answer.replace("*", "")  # Remove any asterisks (bolding)
        answer = answer.strip().lower()
        if not answer:
            print(colored("WARNING in extract_verifier_approval: normalized answer is empty; treating as disapproval.", "yellow"))
            return False

        if "true" in answer:
            return True
        elif "false" in answer:
            return False
        else:
            # Check if 'true' or 'false' is in the first word
            print(colored(f"NOTICE in extract_verifier_approval: {answer=} with {type(answer)=} is not 'true' or 'false', "
                        f"checking if the first word contains 'true' or 'false'...", "magenta"))
            answer_parts = answer.split()
            if not answer_parts:
                print(colored("WARNING in extract_verifier_approval: answer.split() is empty; treating as disapproval.", "yellow"))
                return False
            first_word = answer_parts[0]
            if "true" in first_word:
                print(colored(f"\tSuccess. Found 'true' in first_word.lower(): {first_word.lower()}", "magenta"))
                return True
            elif "false" in first_word:
                print(colored(f"\tSuccess. Found 'false' in first_word.lower(): {first_word.lower()}", "magenta"))
                return False
            else:
                print(colored(f"WARNING in extract_verifier_approval: {answer=} with {type(answer)=} is not 'true' or 'false', "
                            f"AND first word does not contain 'true' or 'false. Full verifier_response: "
                            f"\n{'-' * 30}\n{verifier_response}\n{'-' * 30} (WARNING in extract_verifier_approval)\n", "yellow"))
                return False


    def _get_mathvista_structured_reward(self, solution, problem):
        extraction = extract_mathvista_answer(solution, prompt_idx=self.prompt_idx)
        if extraction is None:
            return -1.0

        normalized = normalize_mathvista_extraction(
            extraction=extraction,
            choices=problem["choices"],
            question_type=problem["question_type"],
            answer_type=problem["answer_type"],
            precision=problem["precision"],
            ignore_empty_extractions=True,
        )
        if normalized is None:
            return -1.0

        score = 0.0
        total = 0

        total += 1
        score += 1.0

        if problem["question_type"] == "multi_choice":
            total += 1
            extraction_text = str(extraction).strip()
            valid_letters = [chr(ord("A") + idx) for idx in range(len(problem["choices"] or []))]
            if extraction_text.upper() in valid_letters or normalized in (problem["choices"] or []):
                score += 1.0
            else:
                score -= 1.0
        elif problem["answer_type"] == "integer":
            total += 1
            cleaned = str(extraction).replace(",", "").strip()
            score += 1.0 if re.fullmatch(r"-?\d+", cleaned) else -1.0
        elif problem["answer_type"] == "float":
            total += 1
            cleaned = str(extraction).replace(",", "").strip()
            try:
                float(cleaned)
                precision = int(problem["precision"]) if problem["precision"] is not None else None
                if precision is None:
                    score += 1.0
                elif "." in cleaned and len(cleaned.split(".")[-1]) == precision:
                    score += 1.0
                else:
                    score -= 1.0
            except Exception:
                score -= 1.0
        elif problem["answer_type"] == "list":
            total += 1
            cleaned = str(extraction).strip()
            score += 1.0 if cleaned.startswith("[") and cleaned.endswith("]") else -1.0

        return score / max(total, 1)


    def get_reward(self, question, solution, media=None, problem=None):
        '''
        Get reward from question and solution.

        Args:
            question: str, question
            solution: str, solution
        Returns:
            reward: int, reward
        '''
        verifications = self.get_verifications(question, solution, media=media, problem=problem)
        reward = 0
        reward_list = self.get_reward_list()
        total = 0
        for verifier_name, verifier_approval in verifications.items():
            total += reward_list[verifier_name]
            if verifier_approval:
                print(colored(f"Verifier {verifier_name} approved the solution.", "green"))
            else:
                print(colored(f"Verifier {verifier_name} disapproved the solution.", "red"))
                reward -= reward_list[verifier_name]

        if self.rule_format_string is not None:
            format_approval = self.get_rule_format_verify(solution)
            if format_approval:
                print(colored(f"Verifier Rule Format approved the solution.", "green"))
            else:
                print(colored(f"Verifier Rule Format disapproved the solution.", "red"))
                reward += -2

        llm_reward = reward / max(total, 1)
        combined_reward = llm_reward

        if self.counterfactual_reward_weight > 0 and media is not None and media.get("image") is not None:
            counterfactual_verifications = self.get_verifications(question, solution, media=None, problem=problem)
            counterfactual_reward = 0
            counterfactual_total = 0
            for verifier_name, verifier_approval in counterfactual_verifications.items():
                counterfactual_total += reward_list[verifier_name]
                if not verifier_approval:
                    counterfactual_reward -= reward_list[verifier_name]
            counterfactual_reward = counterfactual_reward / max(counterfactual_total, 1)
            visual_margin_reward = max(min((llm_reward - counterfactual_reward) / 2.0, 1.0), -1.0)
            print(colored(f"Counterfactual Visual Margin Reward: {visual_margin_reward:.4f}", "blue"))
            weight = min(max(self.counterfactual_reward_weight, 0.0), 1.0)
            combined_reward = (1 - weight) * combined_reward + weight * visual_margin_reward

        if (
            self.structured_reward_weight > 0
            and "MathVista" in self.data_name
            and problem is not None
        ):
            structured_reward = self._get_mathvista_structured_reward(solution, problem)
            print(colored(f"Structured MathVista Reward: {structured_reward:.4f}", "cyan"))
            weight = min(max(self.structured_reward_weight, 0.0), 1.0)
            combined_reward = (1 - weight) * combined_reward + weight * structured_reward

        return combined_reward


    def get_rule_format_verify(self, solution):
        """
        Judge whether the answer follow the format rule.

        Args:
            solution: str
        """
        answer_pattern = self.rule_format_string
        matches = list(re.finditer(answer_pattern, solution, re.DOTALL))
        if len(matches) > 0:
            return True
        else:
            return False


    def get_reward_answer_only(self, question, solution, media=None):
        '''
        Get reward based only on answer.

        Args:
            question: str, question
            solution: str, solution
        Returns:
            reward: int, reward

        Note that when using this reward function, you should only use the "answer_correct" verifier
        '''
        verifications = self.get_verifications(question, solution, media=media)
        reward = 0
        reward_list = self.get_reward_list()
        total = 0
        for verifier_name, verifier_approval in verifications.items():
            total += reward_list[verifier_name]
            if verifier_approval:
                print(colored(f"Verifier {verifier_name} approved the solution.", "green"))
            else:
                print(colored(f"Verifier {verifier_name} disapproved the solution.", "red"))
                reward -= reward_list[verifier_name]
        return reward / total


    def get_reward_list(self):
        '''
        get reward list for different verifiers
        '''
        reward_list = {
            "calculation_check": 2,
            "answer_correct": 1,
            "answer_completeness": 2,
            "understanding_check": 1,
        }
        return reward_list
