from methods.baselines.self_consistency import _build_candidate_record, _sample_answer


def best_of_n_visual_generation(
    model,
    tokenizer,
    device,
    input_text,
    prompt_inputs,
    question,
    media,
    problem,
    visual_reward,
    max_new_tokens,
    data_name,
    prompt_idx,
    model_name_or_path,
    num_candidates=4,
    temperature=0.7,
    top_p=0.95,
):
    candidates = []
    best_idx = 0
    best_score = float("-inf")

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
        score = visual_reward.get_reward(question, answer_text, media=media, problem=problem) if visual_reward is not None else 0.0
        record["score"] = float(score)
        candidates.append(record)
        if score > best_score:
            best_score = float(score)
            best_idx = idx

    selected = candidates[best_idx]
    baseline_info = {
        "mode": "best_of_n_visual",
        "num_candidates": len(candidates),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "selected_index": best_idx,
        "best_score": best_score,
        "candidates": candidates,
    }
    return selected["answer_text"], [best_score], baseline_info
