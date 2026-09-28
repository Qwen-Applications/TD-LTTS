from methods.baselines.self_consistency import _build_candidate_record, _sample_answer


def _score_with_visual_reward(question, candidates, media, problem, visual_reward):
    scores = []
    for candidate in candidates:
        answer_text = candidate["answer_text"]
        score = visual_reward.get_reward(question, answer_text, media=media, problem=problem) if visual_reward else 0.0
        scores.append(
            {
                "answer_reward": float(score),
                "total_score": float(score),
            }
        )
    return scores


def reward_only_rerank_generation(
    model,
    tokenizer,
    device,
    input_text,
    prompt_inputs,
    question,
    media,
    problem,
    visual_reward,
    evidence_reward,
    visual_clues,
    crop_evidence,
    original_answer,
    max_new_tokens,
    data_name,
    prompt_idx,
    model_name_or_path,
    num_candidates=4,
    temperature=0.7,
    top_p=0.95,
):
    """Sample candidates and rerank them with the current reward, without latent updates."""

    candidates = [
        _build_candidate_record(
            answer_text=original_answer,
            data_name=data_name,
            prompt_idx=prompt_idx,
            model_name_or_path=model_name_or_path,
            problem=problem,
        )
    ]
    candidates[0]["source"] = "original"

    for idx in range(max(int(num_candidates), 1)):
        answer_text = _sample_answer(
            input_text=input_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_inputs=prompt_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        record = _build_candidate_record(
            answer_text=answer_text,
            data_name=data_name,
            prompt_idx=prompt_idx,
            model_name_or_path=model_name_or_path,
            problem=problem,
        )
        record["source"] = f"sample_{idx}"
        candidates.append(record)

    answer_texts = [candidate["answer_text"] for candidate in candidates]
    if evidence_reward is not None:
        score_infos = evidence_reward.score_candidates(
            question=question,
            answer_texts=answer_texts,
            media=media,
            problem=problem,
            clues=visual_clues,
            crop_evidence=crop_evidence,
            reference_answer=None,
        )
    else:
        score_infos = _score_with_visual_reward(
            question=question,
            candidates=candidates,
            media=media,
            problem=problem,
            visual_reward=visual_reward,
        )

    if len(score_infos) < len(candidates):
        score_infos = list(score_infos) + [{"total_score": float("-inf")}] * (len(candidates) - len(score_infos))

    for candidate, score_info in zip(candidates, score_infos):
        candidate["reward_info"] = score_info
        candidate["score"] = float(score_info.get("total_score", float("-inf")))
        candidate["valid_for_selection"] = candidate.get("prediction") is not None

    selectable = [idx for idx, candidate in enumerate(candidates) if candidate["valid_for_selection"]]
    if not selectable:
        selectable = list(range(len(candidates)))

    best_idx = max(selectable, key=lambda idx: candidates[idx]["score"])
    selected = candidates[best_idx]
    baseline_info = {
        "mode": "reward_only_rerank",
        "num_sampled_candidates": max(int(num_candidates), 1),
        "num_total_candidates": len(candidates),
        "include_original": True,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "selected_index": int(best_idx),
        "selected_source": selected.get("source"),
        "best_score": float(selected["score"]),
        "used_evidence_reward": evidence_reward is not None,
        "used_visual_reward": visual_reward is not None,
        "num_visual_clues": len(visual_clues or []),
        "used_crop_evidence": crop_evidence is not None,
        "candidates": candidates,
    }
    return selected["answer_text"], [float(selected["score"])], baseline_info
