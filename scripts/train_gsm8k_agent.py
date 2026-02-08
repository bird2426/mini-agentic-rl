import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mini_rl.config import GRPOConfig, ModelConfig, TrainConfig
from src.mini_rl.data.gsm8k import load_gsm8k
from src.mini_rl.trainer import Trainer, make_policy_from_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mini-agentic-rl GSM8K agent")
    parser.add_argument("--model_path", type=str, default="./outputs/Qwen2.5-0.5B/sft")
    parser.add_argument("--output_dir", type=str, default="./outputs/mini_gsm8k_rl")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--max_train_samples", type=int, default=512)
    parser.add_argument("--max_eval_samples", type=int, default=128)
    parser.add_argument("--total_epochs", type=int, default=2)
    parser.add_argument("--samples_per_prompt", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--require_tool_call", action="store_true", default=False)
    parser.add_argument("--save_each_epoch", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_cfg = ModelConfig(
        model_path=args.model_path,
        use_lora=args.use_lora,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    grpo_cfg = GRPOConfig(samples_per_prompt=args.samples_per_prompt)
    train_cfg = TrainConfig(
        total_epochs=args.total_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_r=args.lora_r,
        max_samples=args.max_train_samples,
    )

    print("=" * 80)
    print("Mini-Agentic-RL GSM8K Training")
    print("=" * 80)
    print(f"Model: {model_cfg.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Train samples: {args.max_train_samples}")
    print(f"Eval samples: {args.max_eval_samples}")
    print(f"Epochs: {train_cfg.total_epochs}")
    print("=" * 80)

    train_samples = load_gsm8k(args.train_split, max_samples=args.max_train_samples)
    eval_samples = load_gsm8k(args.eval_split, max_samples=args.max_eval_samples)

    policy = make_policy_from_train_config(train_cfg, model_cfg)
    trainer = Trainer(
        policy, model_cfg, grpo_cfg, require_tool_call=args.require_tool_call
    )

    best_em = -1.0
    best_dir = None

    for epoch in range(1, train_cfg.total_epochs + 1):
        metrics = trainer.train_epoch(epoch, train_samples)
        eval_em = trainer.evaluate_exact_match(eval_samples)

        print(
            f"[Epoch {epoch}] loss={metrics.loss:.4f} avg_reward={metrics.avg_reward:.4f} "
            f"train_em={metrics.exact_match:.4f} eval_em={eval_em:.4f}"
        )

        if args.save_each_epoch:
            epoch_dir = Path(args.output_dir) / f"epoch_{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            policy.save(str(epoch_dir))

        if eval_em > best_em:
            best_em = eval_em
            best_dir = Path(args.output_dir) / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            policy.save(str(best_dir))

    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save(str(final_dir))
    policy.unload()

    print("=" * 80)
    print("Training done")
    print(f"Best eval EM: {best_em:.4f}")
    print(f"Best ckpt: {best_dir}")
    print(f"Final ckpt: {final_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
