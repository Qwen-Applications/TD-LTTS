from methods.latentgrad.optimize import optimized_generation


def legacy_latentseek_generation(
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
    max_new_tokens=1024,
    lr=0.05,
    grad_clip=None,
    k=0.5,
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
        max_num_steps=1,
        max_new_tokens=max_new_tokens,
        lr=lr,
        grad_clip=grad_clip,
        k=k,
        reward_threshold=1e9,
        latent_reg_lambda=0.0,
        update_position_strategy=update_position_strategy,
    )

    baseline_info = {
        "mode": "legacy_latentseek",
        "max_num_steps": 1,
        "reward_model": "legacy_verifier_reward",
    }
    return optimized_output, reward_history, original_length, optimized_length, update_length, baseline_info
