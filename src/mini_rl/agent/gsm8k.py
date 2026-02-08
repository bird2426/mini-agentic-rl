from dataclasses import dataclass
from typing import Dict, Any
import torch

from .base import LitAgent
from ..types import Rollout, Attempt
from ..config import ModelConfig
from ..pipeline import (
    build_gsm8k_prompt,
    rollout_once,
)  # We will refactor pipeline logic later or import from here


@dataclass
class GSM8KAgent(LitAgent):
    model: Any  # HuggingFace model
    tokenizer: Any  # HuggingFace tokenizer
    model_config: ModelConfig

    def rollout(self, rollout: Rollout, attempt: Attempt) -> Dict[str, Any]:
        question = str(rollout.input["question"])
        qid = str(rollout.input["qid"])

        # Domain logic: Build prompt
        prompt = build_gsm8k_prompt(question)

        # Domain logic: Inference
        # Note: rollout_once is currently in pipeline.py.
        # In a full refactor, we might move that logic here or keep it as a utility.
        # For now, we reuse the existing pipeline function to minimize breakage.
        out = rollout_once(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            config=self.model_config,
            group_id=qid,
            do_sample=rollout.mode == "train",
        )

        return {
            "prompt": prompt,
            "response": out.response,
            "tokens": out.tokens,
            "loss_mask": out.loss_mask,
            "group_id": qid,
            "ground_truth": str(rollout.input["ground_truth"]),
        }
