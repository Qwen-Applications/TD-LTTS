"""
Prompt builders for visual clue generation and scoring.
"""

from benchmarks.hallusionbench import build_hallusionbench_question_text
from benchmarks.logicvista import build_logicvista_question_text
from benchmarks.mmstar import build_mmstar_question_text
from benchmarks.mmmu import build_mmmu_question_text
from benchmarks.mathvista import build_mathvista_question_text
from benchmarks.realworldqa import build_realworldqa_question_text
from benchmarks.scienceqa import build_scienceqa_question_text


def _resolve_primary_image(media):
    if not media:
        return None
    image = media.get("image")
    if image is not None:
        return image
    images = media.get("images") or []
    if images:
        return images[0]
    return None


def build_visual_clue_question_text(problem):
    problem = problem or {}
    dataset_name = problem.get("dataset_name")
    if dataset_name == "MMStar":
        return build_mmstar_question_text(problem)
    if dataset_name == "MMMU":
        return build_mmmu_question_text(problem)
    if dataset_name == "LogicVista":
        return build_logicvista_question_text(problem)
    if dataset_name == "RealWorldQA":
        return build_realworldqa_question_text(problem)
    if dataset_name == "HallusionBench":
        return build_hallusionbench_question_text(problem)
    if dataset_name == "ScienceQA":
        return build_scienceqa_question_text(problem)
    return build_mathvista_question_text(problem)


def build_visual_clue_request_text(problem, num_clues=3, draft_response=None):
    lines = [
        build_visual_clue_question_text(problem),
        "Look at the image carefully and list the key visual clues needed to solve the question.",
        f"Return at most {int(num_clues)} clues.",
        "Each clue must be a short atomic fact directly visible in the image.",
        "Do not infer the final answer.",
        "Do not provide reasoning or explanations.",
        'Return each clue on its own line starting with "- ".',
    ]
    if draft_response:
        lines.extend(
            [
                "Draft response for context only:",
                str(draft_response).strip(),
                "Use the draft only to guess what evidence may matter. Do not trust it if it conflicts with the image.",
            ]
        )
    return "\n".join(line for line in lines if line)


def build_visual_clue_support_text(problem, candidate_answer, num_clues=3):
    candidate_answer = str(candidate_answer or "").strip() or "UNKNOWN"
    return "\n".join(
        [
            build_visual_clue_question_text(problem),
            f"Candidate answer: {candidate_answer}",
            "Look at the image carefully.",
            "Return only the short visual clues that are directly visible in the image and support the candidate answer.",
            f"Return at most {int(num_clues)} clues.",
            'Each clue must be on its own line starting with "- ".',
            "Do not add any reasoning, summary, or extra text.",
        ]
    )


def build_visual_clue_request_messages(problem, media=None, num_clues=3, draft_response=None, include_image=True):
    user_content = []
    image = _resolve_primary_image(media)
    if include_image and image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append(
        {
            "type": "text",
            "text": build_visual_clue_request_text(
                problem=problem,
                num_clues=num_clues,
                draft_response=draft_response,
            ),
        }
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a precise visual grounding assistant. Read the image carefully and "
                "return only directly visible clues."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def build_visual_clue_support_messages(problem, media=None, candidate_answer=None, num_clues=3, include_image=True):
    user_content = []
    image = _resolve_primary_image(media)
    if include_image and image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append(
        {
            "type": "text",
            "text": build_visual_clue_support_text(
                problem=problem,
                candidate_answer=candidate_answer,
                num_clues=num_clues,
            ),
        }
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a precise visual grounding assistant. Read the image carefully and "
                "return only directly visible clues that support the candidate answer."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
