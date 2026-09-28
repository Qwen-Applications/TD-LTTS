"""
Solver prompts for different math datasets.
"""
from benchmarks.hallusionbench import hallusionbench_prompt
from benchmarks.logicvista import logicvista_prompt
from benchmarks.mmstar import mmstar_prompt
from benchmarks.mmmu import mmmu_prompt
from benchmarks.mathvista import mathvista_prompt
from benchmarks.realworldqa import realworldqa_prompt
from benchmarks.scienceqa import scienceqa_prompt

def gsm8k_prompt(q, prompt_idx=0):
    """
    Args:
        q (str): The question to be solved.
        prompt_idx (int): The index of the prompt to be used.

    """
    prompt = [
            # idx 0: boxed
            [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": q},
                ],

            # idx 1: json
            [
                {"role": "system", "content": "You are a precise math question solver. Solve this math problem. "},
                {"role": "user", "content":
                 f"QUESTION: {q} \n"
                 "Let's think step by step. "
                 "Please provide your thought process and your final answer separately and response in json format "
                 "containing the keys \"thought process\" and \"final answer\". "
                 "For example your response should be "
                 "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
                 "Note that the final answer should be pure numbers, not the calculation formulas, and without any units or explanation!!! "}
                ],

    ]
    return prompt[prompt_idx]


def MATH_500_prompt(q, prompt_idx=0):
    """
    Args:
        q (str): The question to be solved.
    """
    prompt = [
            # idx 0: boxed
            [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": q},
                ],
            # idx 1: json
            [
                {"role": "system", "content": "You are a precise math question solver. Solve this math problem. "},
                {"role": "user", "content":
                 f"QUESTION: {q} \n"
                 "Let's think step by step. "
                 "Please provide your thought process and your final answer separately and response in json format "
                 "containing the keys \"thought process\" and \"final answer\". "
                 "For example your response should be "
                 "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
                 }
                ],
            ]
    return prompt[prompt_idx]


def AIME_2024_prompt(q, prompt_idx=0):
    """
    Args:
        q (str): The question to be solved.
        prompt_idx (int): The index of the prompt to be used.

    """
    prompt = [
            # idx 0: boxed
            [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": q},
                ],
            # idx 1: json
            [
                {"role": "system", "content": "You are a precise math question solver. Solve this math problem. "},
                {"role": "user", "content":
                 f"QUESTION: {q} \n"
                 "Let's think step by step. "
                 "Please provide your thought process and your final answer separately and response in json format "
                 "containing the keys \"thought process\" and \"final answer\". "
                 "For example your response should be "
                 "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
                 "Note that the final answer should be pure numbers, not the calculation formulas, and without any units or explanation!!! "}
                ],


    ]
    return prompt[prompt_idx]


def MathVista_prompt(problem, image, prompt_idx=1):
    """
    Args:
        problem (dict): one MathVista sample
        image: PIL image
        prompt_idx (int): prompt template index
    """
    return mathvista_prompt(problem, image=image, prompt_idx=prompt_idx)


def MMStar_prompt(problem, image, prompt_idx=1):
    """
    Args:
        problem (dict): one MMStar sample
        image: PIL image
        prompt_idx (int): prompt template index
    """
    return mmstar_prompt(problem, image=image, prompt_idx=prompt_idx)


def MMMU_prompt(problem, image, prompt_idx=1):
    """
    Args:
        problem (dict): one MMMU sample
        image: PIL image
        prompt_idx (int): prompt template index
    """
    return mmmu_prompt(problem, image=image, prompt_idx=prompt_idx)


def LogicVista_prompt(problem, image, prompt_idx=1):
    """
    Args:
        problem (dict): one LogicVista sample
        image: PIL image
        prompt_idx (int): prompt template index
    """
    return logicvista_prompt(problem, image=image, prompt_idx=prompt_idx)


def RealWorldQA_prompt(problem, image, prompt_idx=1):
    """
    Args:
        problem (dict): one RealWorldQA sample
        image: PIL image
        prompt_idx (int): prompt template index
    """
    return realworldqa_prompt(problem, image=image, prompt_idx=prompt_idx)


def HallusionBench_prompt(problem, image, prompt_idx=1):
    """
    Args:
        problem (dict): one HallusionBench sample
        image: PIL image or None for non-image split
        prompt_idx (int): prompt template index
    """
    return hallusionbench_prompt(problem, image=image, prompt_idx=prompt_idx)


def ScienceQA_prompt(problem, image, prompt_idx=1):
    """
    Args:
        problem (dict): one ScienceQA-IMG sample
        image: PIL image
        prompt_idx (int): prompt template index
    """
    return scienceqa_prompt(problem, image=image, prompt_idx=prompt_idx)
