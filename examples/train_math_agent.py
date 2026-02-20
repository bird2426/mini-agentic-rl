"""
Train Math Agent - 展示 Agent-Lightning 正确训练方式

使用 Trainer + MathAgent + Hooks + MiniGRPO 完整训练流程
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_agent_lighting.algorithm.mini_grpo import MiniGRPO
from mini_agent_lighting.litagent import LitAgent
from mini_agent_lighting.store.memory import InMemoryLightningStore
from mini_agent_lighting.trainer import Trainer
from mini_agent_lighting.tracer.dummy import DummyTracer
from mini_agent_lighting.types import Hook, NamedResources, Rollout, Task
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


COT_PROMPT = """Solve the following math problem step by step.

Example 1:
Problem: There are 5 birds on a tree. 2 fly away. How many birds are left?
Solution: 5 - 2 = 3. There are 3 birds left. #### 3

Example 2:
Problem: Tom has 10 apples. He gives 3 to Jerry. How many apples does Tom have now?
Solution: 10 - 3 = 7. Tom has 7 apples now. #### 7

Example 3:
Problem: Lisa has 8 candies. She buys 5 more. How many candies does she have in total?
Solution: 8 + 5 = 13. Lisa has 13 candies in total. #### 13

Now solve this problem:
Problem: {problem}

Solution:"""


class MathTaskInput(BaseModel):
    question: str
    answer: str


class MathAgent(LitAgent[MathTaskInput]):
    """数学问题求解 Agent"""

    def __init__(
        self, model=None, tokenizer=None, prompt_template: str = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template or COT_PROMPT

    def _build_prompt(self, task_input: MathTaskInput) -> str:
        return self.prompt_template.format(problem=task_input.question)

    def _extract_answer(self, response: str) -> Optional[str]:
        response = response.strip()

        if "####" in response:
            for line in reversed(response.split("\n")):
                if "####" in line:
                    return line.split("####")[-1].strip()

        response = re.sub(r"```[\s\S]*?```", "", response)

        match = re.search(
            r"(?:final answer|answer is|result is|=|:)\s*(\d+)", response, re.IGNORECASE
        )
        if match:
            return match.group(1)

        match = re.search(r"\d+\s*[-+\*/]\s*\d+\s*=\s*(\d+)", response)
        if match:
            return match.group(1)

        numbers = re.findall(r"\d+", response)
        if numbers:
            return numbers[-1]

        return None

    def _compute_reward(self, response: str, ground_truth: str) -> float:
        pred = self._extract_answer(response)
        if pred is None:
            return 0.0

        try:
            pred_num = int(pred.replace(",", "").strip())
            truth_num = int(ground_truth.replace(",", "").strip())
            if pred_num == truth_num:
                return 1.0
            elif abs(pred_num - truth_num) <= 1:
                return 0.5
        except:
            pass

        return 0.0

    def rollout(
        self,
        task: Task,
        resources: NamedResources,
        rollout: Rollout,
    ) -> Optional[float]:
        task_input = task.input
        prompt = self._build_prompt(task_input)

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        rollout.output = response

        answer_str = (
            task_input.answer.split("####")[-1].strip()
            if "####" in task_input.answer
            else task_input.answer
        )
        reward = self._compute_reward(response, answer_str)
        rollout.metadata["reward"] = reward

        return reward


class LoggingHook(Hook):
    """日志 Hook - 记录训练过程"""

    async def on_rollout_start(
        self,
        agent: LitAgent,
        runner: Any,
        rollout: Rollout,
    ) -> None:
        logger.info(f"Starting rollout")

    async def on_rollout_end(
        self,
        agent: LitAgent,
        runner: Any,
        rollout: Rollout,
        spans: Any,
    ) -> None:
        reward = rollout.metadata.get("reward", 0.0)
        logger.info(f"Rollout complete. Reward: {reward}")


def load_gsm8k_train(num_samples: int = 50):
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    if num_samples:
        dataset = dataset.select(range(num_samples))
    return dataset


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--train_samples", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Loading model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    logger.info("Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    logger.info(f"Loading GSM8K train data ({args.train_samples} samples)...")
    train_data = load_gsm8k_train(num_samples=args.train_samples)

    agent = MathAgent(model=model, tokenizer=tokenizer)

    logger.info(f"Training for {args.iterations} iterations...")

    grpo = MiniGRPO(
        model=model,
        tokenizer=tokenizer,
        lr=args.lr,
        batch_size=1,
        epochs=3,
    )

    for iteration in range(args.iterations):
        prompts = []
        responses = []
        rewards = []

        for item in train_data:
            task_input = MathTaskInput(question=item["question"], answer=item["answer"])
            prompt = agent._build_prompt(task_input)

            answer_str = (
                item["answer"].split("####")[-1].strip()
                if "####" in item["answer"]
                else item["answer"]
            )

            device = next(model.parameters()).device
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                )
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            reward = agent._compute_reward(response, answer_str)

            prompts.append(prompt)
            responses.append(prompt + response)
            rewards.append(reward)

        if iteration % 2 == 0:
            logger.info(f"Iteration {iteration}: rewards = {rewards[:4]}...")

        result = grpo.fit(prompts, responses, rewards)

    logger.info("Training complete!")

    logger.info("\n要评估模型,请运行:")
    logger.info(
        "python examples/benchmark.py --model Qwen/Qwen2-0.5B --adapter ./lora_adapter"
    )


if __name__ == "__main__":
    main()
