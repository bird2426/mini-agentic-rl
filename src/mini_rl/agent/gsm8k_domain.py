import re
from dataclasses import dataclass
from typing import Any, List, Dict, Optional
import torch
from ..config import ModelConfig


@dataclass
class RolloutSample:
    group_id: str
    prompt: str
    response: str
    tokens: List[int]
    loss_mask: List[int]
    metadata: Dict[str, Any]


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


# Reward Logic
def normalize_numeric(s: str) -> str:
    # Remove all non-numeric characters (keep dots for decimals)
    s = "".join(c for c in s if c.isdigit() or c == ".")
    if not s:
        return ""
    try:
        # Normalize: "1000" -> "1000", "1,000" -> "1000", "1.0" -> "1"
        return str(float(s)).rstrip("0").rstrip(".")
    except ValueError:
        return ""


def extract_final_answer(text: str) -> Optional[str]:
    # Look for "#### <answer>" pattern
    parts = text.split("####")
    if len(parts) < 2:
        return None
    candidate = parts[-1].strip()
    return normalize_numeric(candidate)


def has_tool_call_markup(response: str) -> bool:
    # Dummy tool call check for compatibility
    return "<tool_code>" in response


def score_gsm8k_response(
    response: str, ground_truth: str, require_tool_call: bool = False
) -> float:
    if require_tool_call and not has_tool_call_markup(response):
        return 0.0

    prediction = extract_final_answer(response)
    if prediction is None:
        return 0.0

    gt_norm = normalize_numeric(ground_truth)
    if not gt_norm:
        # If GT is invalid/empty, we can't score properly. Default to 0.
        return 0.0

    return 1.0 if prediction == gt_norm else 0.0
