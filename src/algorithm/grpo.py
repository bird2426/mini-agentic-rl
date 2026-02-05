from __future__ import annotations
import logging
import asyncio
from typing import Any, Dict, List, Optional
from src.core.types import Triplet, Rollout, Span
from src.store.memory import InMemoryStore
from src.trainer import RLTrainer
from src.core.adapter import TraceToTripletAdapter

logger = logging.getLogger(__name__)

class BaseAlgorithm:
    def __init__(self, store: InMemoryStore):
        self.store = store

    async def run(self):
        raise NotImplementedError()

class GRPOAlgorithm(BaseAlgorithm):
    def __init__(self, store: InMemoryStore, trainer_config: Dict[str, Any], tokenizer: Any):
        super().__init__(store)
        self.trainer = RLTrainer(trainer_config)
        self.tokenizer = tokenizer
        self.adapter = TraceToTripletAdapter()
        self.current_model_path = trainer_config.get("model_path")

    def _tokenize_triplet(self, triplet: Triplet) -> Dict[str, Any]:
        prompt_text = self.tokenizer.apply_chat_template(triplet.prompt, tokenize=False, add_generation_prompt=True)
        response_text = triplet.response
        
        full_text = prompt_text + response_text
        tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        
        loss_mask = [0] * len(prompt_tokens) + [1] * (len(tokens) - len(prompt_tokens))
        
        return {
            "tokens": tokens,
            "loss_mask": loss_mask,
            "reward": triplet.reward
        }

    async def run_step(self, prompts: List[str], samples_per_prompt: int, epoch: int):
        rollout_ids = []
        for prompt in prompts:
            for _ in range(samples_per_prompt):
                rollout = await self.store.enqueue_rollout(prompt)
                rollout_ids.append(rollout.rollout_id)
        
        logger.info(f"Enqueued {len(rollout_ids)} rollouts")
        await self.store.wait_for_rollouts(rollout_ids)
        
        all_triplets = []
        for rid in rollout_ids:
            spans = await self.store.get_spans(rid)
            triplets = self.adapter.adapt(spans)
            all_triplets.extend(triplets)
            
        # Group-Relative Advantage Calculation
        num_prompts = len(prompts)
        trajectories_for_training = []

        for i in range(num_prompts):
            prompt_triplets = all_triplets[i * samples_per_prompt : (i+1) * samples_per_prompt]
            rewards = [t.reward for t in prompt_triplets]

            logger.info(f"[GRPO] Prompt {i}: rewards={rewards}, unique rewards: {set(rewards)}")
            logger.info(f"[GRPO] Sample responses:")
            for j, t in enumerate(prompt_triplets):
                response_preview = t.response[:100] if len(t.response) > 100 else t.response
                logger.info(f"  [{j}] reward={t.reward}: {response_preview}...")

            import torch
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            mean_reward = rewards_tensor.mean()
            std_reward = rewards_tensor.std() + 1e-8

            for t in prompt_triplets:
                traj = self._tokenize_triplet(t)
                advantage = (t.reward - mean_reward.item()) / std_reward.item()
                if torch.isnan(torch.tensor(advantage)):
                    advantage = 0.0
                traj["advantage"] = advantage
                trajectories_for_training.append(traj)

        logger.info(f"Starting training with {len(trajectories_for_training)} trajectories")
        self.trainer.load_model(self.current_model_path)
        self.trainer.train(trajectories_for_training)
        
        output_dir = f"outputs/epoch_{epoch}"
        self.trainer.save_model(output_dir)
        self.trainer.unload_model()
        self.current_model_path = output_dir
        
        return trajectories_for_training
