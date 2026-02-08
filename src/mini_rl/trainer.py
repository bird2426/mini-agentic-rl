from dataclasses import dataclass
from statistics import mean
from typing import List, Dict, Any

from .config import GRPOConfig, ModelConfig, TrainConfig
from .data.gsm8k import GSM8KSample
from .model.policy import PolicyModel
from .types import AlgorithmMetrics

# New components
from .store.memory import InMemoryStore
from .runner.local import LocalRunner
from .agent.gsm8k import GSM8KAgent
from .agent.gsm8k_domain import score_gsm8k_response
from .adapter.trace import TraceAdapter
from .algorithm.grpo import LocalGRPOAlgorithm


@dataclass
class EpochMetrics:
    epoch: int
    avg_reward: float
    exact_match: float
    loss: float
    num_trajectories: int


class Trainer:
    def __init__(
        self,
        policy: PolicyModel,
        model_config: ModelConfig,
        grpo_config: GRPOConfig,
        require_tool_call: bool = False,
    ):
        self.policy = policy
        self.model_config = model_config
        self.grpo_config = grpo_config

        # Hub: Store
        self.store = InMemoryStore()

        # Spoke: Algorithm (connected to Policy and Adapter)
        # We inject the domain-specific reward function here
        self.adapter = TraceAdapter(
            reward_fn=score_gsm8k_response, require_tool_call=require_tool_call
        )
        self.algorithm = LocalGRPOAlgorithm(policy=self.policy, adapter=self.adapter)

        # Initial resources
        self.store.add_resources({"policy": self.policy})

        # Spoke: Agent
        self.agent = GSM8KAgent(
            model=self.policy.model,
            tokenizer=self.policy.tokenizer,
            model_config=self.model_config,
        )

        # Spoke: Runner (connects Store and Agent)
        self.runner = LocalRunner(store=self.store, agent=self.agent)

    def train_epoch(self, epoch: int, samples: List[GSM8KSample]) -> EpochMetrics:
        for sample in samples:
            payload: Dict[str, Any] = {
                "qid": sample.qid,
                "question": sample.question,
                "ground_truth": sample.ground_truth,
            }
            for _ in range(self.grpo_config.samples_per_prompt):
                self.store.enqueue_rollout(payload, mode="train")

        self.runner.run_until_empty()

        completed = self.store.query_rollouts(status_in=("succeeded",))
        span_records: List[Dict[str, Any]] = []
        for rollout in completed:
            spans = self.store.query_spans(rollout.rollout_id)
            span_records.extend(
                [
                    {
                        "rollout_id": span.rollout_id,
                        "name": span.name,
                        "payload": span.payload,
                    }
                    for span in spans
                ]
            )

        triplets = self.algorithm.build_trajectories(span_records)
        metrics: AlgorithmMetrics = self.algorithm.optimize(triplets)

        return EpochMetrics(
            epoch=epoch,
            avg_reward=metrics.avg_reward,
            exact_match=metrics.exact_match,
            loss=metrics.loss,
            num_trajectories=metrics.trajectories,
        )

    def evaluate_exact_match(self, samples: List[GSM8KSample]) -> float:
        # Isolated evaluation environment
        eval_store = InMemoryStore()
        eval_store.add_resources({"policy": self.policy})

        eval_agent = GSM8KAgent(
            model=self.policy.model,
            tokenizer=self.policy.tokenizer,
            model_config=self.model_config,
        )

        eval_runner = LocalRunner(store=eval_store, agent=eval_agent)

        # Wire evaluation adapter with the same reward function
        adapter = TraceAdapter(reward_fn=score_gsm8k_response, require_tool_call=False)

        # Enqueue evaluation tasks
        for sample in samples:
            eval_store.enqueue_rollout(
                {
                    "qid": sample.qid,
                    "question": sample.question,
                    "ground_truth": sample.ground_truth,
                },
                mode="eval",
            )

        eval_runner.run_until_empty()

        completed = eval_store.query_rollouts(status_in=("succeeded",))
        all_triplets = []
        for rollout in completed:
            spans = eval_store.query_spans(rollout.rollout_id)
            span_records = [
                {
                    "rollout_id": span.rollout_id,
                    "name": span.name,
                    "payload": span.payload,
                }
                for span in spans
            ]
            all_triplets.extend(adapter.adapt(span_records))

        exact = [1.0 if t.reward >= 1.0 else 0.0 for t in all_triplets]
        return mean(exact) if exact else 0.0


def make_policy_from_train_config(
    train_cfg: TrainConfig, model_cfg: ModelConfig
) -> PolicyModel:
    trainer_config = {
        "learning_rate": train_cfg.learning_rate,
        "batch_size": train_cfg.batch_size,
        "gradient_accumulation_steps": train_cfg.gradient_accumulation_steps,
        "lora_r": train_cfg.lora_r,
        "lora_alpha": train_cfg.lora_r * 2,
    }
    return PolicyModel.create(
        trainer_config, model_cfg.model_path, use_lora=model_cfg.use_lora
    )
