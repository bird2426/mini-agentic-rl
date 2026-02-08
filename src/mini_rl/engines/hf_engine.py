import os
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..algorithm.math import group_normalize_advantages


class HFBackend:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.optimizer = None

    def load_model(self, model_path: str, use_lora: bool = True) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        base_model_path = model_path
        is_peft_checkpoint = os.path.exists(
            os.path.join(model_path, "adapter_config.json")
        )

        if is_peft_checkpoint:
            from peft import PeftConfig, PeftModel

            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_path = peft_config.base_model_name_or_path
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(
                base_model, model_path, is_trainable=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            if use_lora:
                self.model.gradient_checkpointing_enable()
                self.model = prepare_model_for_kbit_training(self.model)
                lora_config = LoraConfig(
                    r=self.config.get("lora_r", 8),
                    lora_alpha=self.config.get("lora_alpha", 16),
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                    lora_dropout=self.config.get("lora_dropout", 0.1),
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.model = get_peft_model(self.model, lora_config)

        tokenizer_path = base_model_path if is_peft_checkpoint else model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def update(self, trajectories: list[dict[str, Any]]) -> dict[str, Any]:
        if not trajectories:
            return {"loss": 0.0}
        metrics = self.train(trajectories, num_epochs=1)
        return {"loss": metrics.get("avg_loss", 0.0), **metrics}

    def train(
        self, trajectories: list[dict[str, Any]], num_epochs: int = 1
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if self.optimizer is None:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(), lr=self.config.get("learning_rate", 1e-5)
            )

        batch_size = self.config.get("batch_size", 1)
        grad_accum_steps = self.config.get("gradient_accumulation_steps", 4)
        last_avg_epoch_loss = 0.0

        for _ in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            for i in range(0, len(trajectories), batch_size):
                batch = trajectories[i : i + batch_size]
                batch_tokens, batch_masks, batch_rewards, group_ids = (
                    self._prepare_batch(batch)
                )
                loss = self._compute_loss_with_mask(
                    batch_tokens, batch_masks, batch_rewards, group_ids
                )
                loss = loss / grad_accum_steps
                loss.backward()
                if (i // batch_size + 1) % grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                epoch_loss += loss.item() * grad_accum_steps
                num_batches += 1
            if num_batches:
                last_avg_epoch_loss = epoch_loss / num_batches

        return {"avg_loss": last_avg_epoch_loss, "num_trajectories": len(trajectories)}

    def _prepare_batch(
        self, batch: list[dict[str, Any]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        max_len = min(256, max(len(traj["tokens"]) for traj in batch))
        batch_tokens = []
        batch_masks = []
        batch_rewards = []
        group_ids: list[str] = []
        for traj in batch:
            tokens = traj["tokens"][:max_len]
            mask = traj["loss_mask"][:max_len]
            reward = traj["reward"]
            group_ids.append(str(traj.get("group_id", "default")))
            pad_len = max_len - len(tokens)
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            if pad_id is None:
                raise RuntimeError("Tokenizer must define pad token or eos token")
            batch_tokens.append(tokens + [pad_id] * pad_len)
            batch_masks.append(mask + [0] * pad_len)
            batch_rewards.append(reward)

        return (
            torch.tensor(batch_tokens, dtype=torch.long),
            torch.tensor(batch_masks, dtype=torch.float32),
            torch.tensor(batch_rewards, dtype=torch.float32),
            group_ids,
        )

    def _compute_loss_with_mask(
        self,
        tokens: torch.Tensor,
        loss_masks: torch.Tensor,
        rewards: torch.Tensor,
        group_ids: list[str],
    ) -> torch.Tensor:
        device = next(self.model.parameters()).device
        tokens = tokens.to(device)
        loss_masks = loss_masks.to(device)
        rewards = rewards.to(device)
        outputs = self.model(
            input_ids=tokens[:, :-1], labels=tokens[:, 1:], use_cache=False
        )
        logits = outputs.logits
        vocab_size = logits.size(-1)
        shift_labels = tokens[:, 1:].contiguous()
        shift_masks = loss_masks[:, 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fct(
            logits.view(-1, vocab_size), shift_labels.view(-1)
        ).view_as(shift_labels)
        masked_loss = token_loss * shift_masks
        per_sample_loss = masked_loss.sum(dim=1) / shift_masks.sum(dim=1).clamp_min(1)
        advantages = group_normalize_advantages(rewards, group_ids).to(device)
        return (per_sample_loss * advantages).mean()

    def save_model(self, output_dir: str) -> None:
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def unload_model(self) -> None:
        del self.model
        self.model = None
        torch.cuda.empty_cache()
