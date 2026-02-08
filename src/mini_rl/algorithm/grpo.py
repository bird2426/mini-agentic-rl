from typing import List, Any, Dict
from statistics import mean

from .base import Algorithm
from ..types import Triplet, AlgorithmMetrics
from ..adapter.base import Adapter


class LocalGRPOAlgorithm(Algorithm):
    def __init__(self, policy: Any, adapter: Adapter):
        self.policy = policy
        self.adapter = adapter

    def optimize(self, triplets: List[Triplet]) -> AlgorithmMetrics:
        # Convert triplets to format expected by PolicyModel.update
        trajectories = [
            {
                "tokens": item.token_ids,
                "loss_mask": item.loss_mask,
                "reward": item.reward,
                "group_id": item.group_id,
            }
            for item in triplets
        ]

        # Policy update
        update = self.policy.update(trajectories)

        # Calculate metrics
        rewards = [t.reward for t in triplets]
        exact = [1.0 if t.reward >= 1.0 else 0.0 for t in triplets]

        return AlgorithmMetrics(
            loss=float(update.get("loss", 0.0)),
            avg_reward=mean(rewards) if rewards else 0.0,
            exact_match=mean(exact) if exact else 0.0,
            trajectories=len(trajectories),
        )

    def build_trajectories(self, span_records: List[Dict[str, Any]]) -> List[Triplet]:
        """Helper to use the adapter directly if needed."""
        return self.adapter.adapt(span_records)
