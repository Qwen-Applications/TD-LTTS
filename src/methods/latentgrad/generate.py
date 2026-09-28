import torch

from methods.latentgrad.generation_utils import (
    get_stop_token_ids,
    move_model_inputs,
    reset_multimodal_generation_state,
    trim_after_stop_words,
)

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]


def original_generation(input_text, model, tokenizer, device, prompt_inputs=None, max_new_tokens=1024):
    '''
    Generate answer using the model's native generation path.

    Returns:
        answer: original generated answer
        hidden_states_list: list of last-layer hidden states for each generated token step
        new_input_ids: prompt ids concatenated with generated non-stop token ids
    '''
    eos_token = tokenizer.eos_token
    if eos_token is not None and eos_token not in stop_words:
        stop_words.append(eos_token)

    if prompt_inputs is None:
        inputs = tokenizer([input_text], return_tensors="pt").to(device)
    else:
        inputs = move_model_inputs(prompt_inputs, device)

    prompt_input_ids = inputs["input_ids"].clone()
    prompt_length = prompt_input_ids.shape[-1]

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    stop_token_ids = get_stop_token_ids(tokenizer)
    reset_multimodal_generation_state(model)
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    generated_token_ids = generation_output.sequences[0][prompt_length:].tolist()
    hidden_states_list = []
    answer_token_ids = []
    generated_ids_nonstop = []

    for step_idx, step_hidden_states in enumerate(generation_output.hidden_states):
        if step_idx >= len(generated_token_ids):
            break
        token_id = generated_token_ids[step_idx]
        hidden_states_list.append(step_hidden_states[-1][:, -1].clone())
        answer_token_ids.append(token_id)

        new_token = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if token_id == tokenizer.eos_token_id or new_token in stop_words:
            break
        generated_ids_nonstop.append(token_id)

    if generated_ids_nonstop:
        generated_tensor = torch.tensor([generated_ids_nonstop], dtype=torch.long, device=device)
        new_input_ids = torch.cat([prompt_input_ids, generated_tensor], dim=-1)
    else:
        new_input_ids = prompt_input_ids

    answer = tokenizer.decode(
        answer_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    answer = trim_after_stop_words(answer, stop_words)
    return answer, hidden_states_list, new_input_ids
