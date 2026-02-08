from dataclasses import dataclass
from datasets import load_dataset

from ..reward import normalize_numeric


@dataclass
class GSM8KSample:
    qid: str
    question: str
    answer: str
    ground_truth: str


def load_gsm8k(split: str, max_samples: int | None = None) -> list[GSM8KSample]:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    items: list[GSM8KSample] = []
    for idx, row in enumerate(ds):
        row_dict = dict(row)
        answer = str(row_dict.get("answer", ""))
        gt = answer.split("####")[-1].strip() if "####" in answer else answer
        items.append(
            GSM8KSample(
                qid=f"{split}-{idx}",
                question=str(row_dict.get("question", "")),
                answer=answer,
                ground_truth=normalize_numeric(gt),
            )
        )
    return items
