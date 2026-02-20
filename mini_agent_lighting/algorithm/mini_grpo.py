import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mini_agent_lighting.algorithm.base import Algorithm
from mini_agent_lighting.client import AgentLightningClient
from mini_agent_lighting.types import Dataset as ALGDataset

logger = logging.getLogger(__name__)


class GRPODataset(Dataset[Any]):
    def __init__(self, prompts: list[str], responses: list[str], rewards: list[float]):
        self.prompts = prompts
        self.responses = responses
        self.rewards = rewards

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "prompt": self.prompts[idx],
            "response": self.responses[idx],
            "reward": self.rewards[idx],
        }


class MiniGRPO(Algorithm):
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        lr: float = 1e-5,
        batch_size: int = 4,
        epochs: int = 3,
        gamma: float = 0.99,
        beta: float = 0.1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.gamma = gamma
        self.beta = beta

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.old_model = None

    def compute_grpo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        per_token_ratio = torch.exp(log_probs - old_log_probs)
        per_sample_ratio = per_token_ratio.mean(dim=1)
        surr1 = per_sample_ratio * advantages
        surr2 = torch.clamp(per_sample_ratio, 1 - self.beta, 1 + self.beta) * advantages
        loss = -torch.min(surr1, surr2).mean()
        return loss

    def compute_advantages(self, rewards: list[float]) -> torch.Tensor:
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean = rewards_tensor.mean()
        std = rewards_tensor.std()
        advantages = (rewards_tensor - mean) / (std + 1e-8)
        return advantages

    def update(
        self, prompts: list[str], responses: list[str], rewards: list[float]
    ) -> dict[str, float]:
        if len(prompts) == 0:
            return {"loss": 0.0}

        device = next(self.model.parameters()).device
        advantages = self.compute_advantages(rewards)

        self.model.train()

        full_inputs = self.tokenizer(
            responses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        prompt_len = prompt_inputs.input_ids.shape[1]

        response_len = full_inputs.input_ids.shape[1] - prompt_len
        if response_len <= 0:
            return {"loss": 0.0}

        with torch.no_grad():
            outputs = self.model(full_inputs.input_ids)
            logits = outputs.logits[
                :, prompt_len - 1 : prompt_len + response_len - 1, :
            ]
            response_tokens = full_inputs.input_ids[
                :, prompt_len : prompt_len + response_len
            ]
            old_log_probs = torch.log_softmax(logits, dim=-1)
            old_log_probs = old_log_probs.gather(
                2, response_tokens.unsqueeze(-1)
            ).squeeze(-1)

        self.optimizer.zero_grad()

        outputs = self.model(full_inputs.input_ids)
        logits = outputs.logits[:, prompt_len - 1 : prompt_len + response_len - 1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs = log_probs.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)

        advantages = advantages.to(device)

        loss = self.compute_grpo_loss(
            log_probs,
            old_log_probs,
            advantages,
        )

        loss.backward()
        self.optimizer.step()

        torch.cuda.empty_cache()

        return {"loss": loss.item()}

    def fit(
        self, prompts: list[str], responses: list[str], rewards: list[float]
    ) -> dict[str, float]:
        total_loss = 0.0
        for epoch in range(self.epochs):
            result = self.update(prompts, responses, rewards)
            total_loss += result.get("loss", 0.0)
            logger.info(
                f"GRPO Epoch {epoch + 1}/{self.epochs}, Loss: {result.get('loss', 0.0):.4f}"
            )
        return {"avg_loss": total_loss / self.epochs}

    def run(
        self,
        train_dataset: Optional[ALGDataset[Any]] = None,
        val_dataset: Optional[ALGDataset[Any]] = None,
    ) -> None:
        logger.info("MiniGRPO.run() called - using standalone fit() for now")
        if train_dataset is not None:
            logger.info(f"Training with dataset of size: {len(train_dataset)}")

    def get_client(self) -> AgentLightningClient:
        raise NotImplementedError("MiniGRPO uses standalone mode, not client-server")

    def set_trainable_params(
        self, prompts: list[str], responses: list[str], rewards: list[float]
    ) -> None:
        pass


__all__ = ["MiniGRPO", "GRPODataset"]
