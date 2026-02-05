"""Mini Agentic RL Training - Single script with YAML config support

Usage:
    python train.py --config config/sft_gsm8k.yaml
    python train.py --config config/grpo_gsm8k.yaml
"""
import argparse
import random
import asyncio
import sys
import logging
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.core.types import NamedResources
from src.store.memory import InMemoryLightningStore
from src.runner.agent import LitAgentRunner
from src.agents.gsm8k_lit import GSM8KLitAgent
from src.algorithm.grpo import GRPOAlgorithm
from src.datasets.gsm8k import GSM8KDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device(device_str: str) -> str:
    """Get device string, supports 'auto' detection"""
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return device_str


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def ensure_numeric(config: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    for key in keys:
        if key in config:
            val = config[key]
            try:
                config[key] = int(val)
            except ValueError:
                config[key] = float(val)
    return config


async def run_sft(config: dict[str, Any], trainer_config: dict[str, Any], tokenizer: Any):
    """Run SFT training mode"""
    logger.info("=" * 50)
    logger.info("Running SFT Training")
    logger.info("=" * 50)

    device = config.get("device", "cpu")
    model_path = config.get("model_path")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("config.model_path must be a non-empty string")

    dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 3e-5))

    dataset = GSM8KDataset()
    train_data = dataset.load("train", max_samples=config.get("max_samples", 100))

    def format_sample(item):
        question = item["question"]
        answer = item["answer"]
        if "####" in answer:
            final = answer.split("####")[-1].strip()
        else:
            final = answer
        messages = [
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"{answer}\n\nThe answer is {final}"}
        ]
        return messages

    model.train()
    batch_size = config.get("batch_size", 2)
    grad_accum = config.get("gradient_accumulation_steps", 4)
    epochs = config.get("num_epochs", 3)

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")

        # Shuffle the list
        shuffled = train_data.copy()
        random.seed(42 + epoch)
        random.shuffle(shuffled)
        total_loss = 0
        num_batches = 0

        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i:i + batch_size]
            messages_list = [format_sample(item) for item in batch]

            texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                     for m in messages_list]
            encoded = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
            input_ids = torch.tensor(encoded["input_ids"]).to(device)
            labels = torch.tensor([x.copy() for x in encoded["input_ids"]]).to(device)
            attention_mask = torch.tensor(encoded["attention_mask"]).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()

            if (i // batch_size + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum
            num_batches += 1

            if num_batches % 10 == 0:
                logger.info(f"  Batch {num_batches}, Loss: {total_loss / num_batches:.4f}")

    output_path = config.get("output_dir", "./outputs")
    logger.info(f"Saving model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


async def run_grpo(config: dict[str, Any], trainer_config: dict[str, Any]):
    """Run GRPO training mode"""
    logger.info("=" * 50)
    logger.info("Running GRPO Training")
    logger.info("=" * 50)

    device = config.get("device", "cpu")
    model_path = config.get("model_path")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("config.model_path must be a non-empty string")

    store = InMemoryLightningStore()

    # Load model and tokenizer
    dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    await store.add_resources({
        "model": model,
        "tokenizer": tokenizer
    })

    agent = GSM8KLitAgent(
        max_turns=config.get("max_turns", 2),
        use_server=False
    )
    runner = LitAgentRunner(tracer=None)
    runner.init(agent=agent)
    runner.init_worker(worker_id="worker-1", store=store)
    runner_task = asyncio.create_task(runner.start())

    algorithm = GRPOAlgorithm(store, trainer_config, tokenizer)

    dataset = GSM8KDataset()
    train_data = dataset.load(config.get("split", "train"), max_samples=config.get("max_samples", 10))

    prompts_with_metadata = []
    for item in train_data:
        prompt = item["question"]
        ground_truth = item["answer"].split("####")[-1].strip() if "####" in item["answer"] else item["answer"]
        prompts_with_metadata.append({
            "prompt": prompt,
            "ground_truth": ground_truth
        })

    samples_per_prompt = config.get("samples_per_prompt", 4)
    num_epochs = config.get("num_epochs", 3)

    logger.info(f"Loaded {len(prompts_with_metadata)} prompts")
    logger.info(f"Training config: epochs={num_epochs}, samples_per_prompt={samples_per_prompt}")

    try:
        for epoch in range(1, num_epochs + 1):
            logger.info(f"=== Epoch {epoch}/{num_epochs} ===")
            trajectories = await algorithm.run_step(
                prompts_with_metadata,
                samples_per_prompt=samples_per_prompt,
                epoch=epoch
            )
            logger.info(f"Collected {len(trajectories)} trajectories")
    finally:
        runner.stop()
        await runner_task


def main():
    parser = argparse.ArgumentParser(description="Mini Agentic RL Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--mode", type=str, default=None, help="Override mode: sft or grpo")
    args = parser.parse_args()

    config = load_config(args.config)
    mode = args.mode or config.get("mode", "grpo")

    numeric_keys = ["learning_rate", "batch_size", "gradient_accumulation_steps",
                    "num_epochs", "max_samples", "samples_per_prompt", "max_turns", "max_tokens"]
    config = ensure_numeric(config, numeric_keys)

    # Resolve device
    device = get_device(config.get("device", "cpu"))
    config["device"] = device
    logger.info(f"Using device: {device}")

    model_path = config.get("model_path")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("config.model_path must be a non-empty string")
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    trainer_config = {
        "model_path": model_path,
        "learning_rate": config.get("learning_rate", 1e-5),
        "batch_size": config.get("batch_size", 1),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 4),
        "lora_r": config.get("lora_r", 8),
        "lora_alpha": config.get("lora_alpha", 16),
    }

    if mode == "sft":
        asyncio.run(run_sft(config, trainer_config, tokenizer))
    elif mode == "grpo":
        asyncio.run(run_grpo(config, trainer_config))
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
