from dataclasses import dataclass
from typing import Any

import torch

from .config import ModelConfig


@dataclass
class RolloutSample:
    group_id: str
    prompt: str
    response: str
    tokens: list[int]
    loss_mask: list[int]
    metadata: dict[str, Any]


def build_gsm8k_prompt(question: str) -> str:
    return (
        "You are a math solver. Provide the final numeric answer using format: #### <number>.\n"
        "Keep reasoning concise.\n\n"
        f"Question: {question}\n"
    )


def rollout_once(
    model: Any,
    tokenizer: Any,
    prompt: str,
    config: ModelConfig,
    group_id: str,
    do_sample: bool = True,
) -> RolloutSample:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = max(config.temperature, 1e-5)
            generation_kwargs["top_p"] = config.top_p

        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )

    input_len = inputs["input_ids"].shape[1]
    response_ids = outputs[0][input_len:].tolist()
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    full_ids = outputs[0].tolist()
    loss_mask = [0] * input_len + [1] * len(response_ids)

    return RolloutSample(
        group_id=group_id,
        prompt=prompt,
        response=response,
        tokens=full_ids,
        loss_mask=loss_mask,
        metadata={"prompt_len": input_len},
    )
