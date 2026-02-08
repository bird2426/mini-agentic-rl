import re
from decimal import Decimal, InvalidOperation


def normalize_numeric(value: str) -> str:
    text = value.strip().replace(",", "")
    if not text:
        return ""
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"-0", "+0", ""}:
        return "0"
    if normalized.startswith("+"):
        normalized = normalized[1:]
    return normalized


def extract_final_answer(text: str) -> str:
    if "####" in text:
        tail = text.split("####")[-1].strip()
        if not tail:
            return ""
        lines = tail.splitlines()
        raw = lines[0].strip() if lines else tail
        return normalize_numeric(raw)
    matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not matches:
        return ""
    return normalize_numeric(matches[-1])


def has_tool_call_markup(text: str) -> bool:
    return (
        "<tool>" in text
        and "</tool>" in text
        and "<args>" in text
        and "</args>" in text
    )


def score_gsm8k_response(
    response: str, ground_truth: str, require_tool_call: bool = False
) -> float:
    pred = extract_final_answer(response)
    target = normalize_numeric(ground_truth)
    correct = pred != "" and pred == target
    used_tool = has_tool_call_markup(response)

    reward = 0.0
    if correct:
        reward += 2.0
    elif pred != "":
        reward += 0.1

    if "####" in response:
        reward += 0.05

    if used_tool:
        reward += 0.05
    elif require_tool_call:
        reward = max(0.0, reward - 0.1)

    return reward
