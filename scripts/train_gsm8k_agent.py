import sys
import os
from pathlib import Path

# 优化显存分配
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from transformers import AutoTokenizer
from src.agents import GSM8KAgent
from src.rollout import RolloutManager
from src.trainer import RLTrainer
from src.datasets import GSM8KDataset


def main():
    parser = argparse.ArgumentParser(description="GSM8K Agent RL 训练")
    
    # 模型
    parser.add_argument("--model_path", type=str, default="./outputs/Qwen2.5-0.5B/sft",
                       help="SFT 模型路径")
    parser.add_argument("--output_dir", type=str, default="./outputs/Qwen2.5-0.5B/rl",
                       help="输出目录")
    
    # 数据
    parser.add_argument("--dataset", type=str, default="gsm8k",
                       choices=["gsm8k"])
    parser.add_argument("--samples_per_prompt", type=int, default=4,
                       help="GRPO: 每个问题采样几次")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="每次生成的最大 token 数")
    
    # 训练
    parser.add_argument("--total_epochs", type=int, default=3,
                       help="训练轮数（每轮: Rollout → Training）")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    
    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=8)
    
    # 调试
    parser.add_argument("--max_samples", type=int, default=None,
                       help="限制样本数（默认 None=全部）")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GSM8K Agent RL 训练")
    print("=" * 80)
    print(f"模型: {args.model_path}")
    print(f"数据: {args.dataset} ({'全部' if args.max_samples is None else f'{args.max_samples} 样本'})")
    print(f"GRPO: 每个问题采样 {args.samples_per_prompt} 次")
    print(f"训练: {args.total_epochs} epochs")
    print("=" * 80)
    
    
    # 初始化
    agent = GSM8KAgent()  # 内置工具
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.dataset == "gsm8k":
        dataset = GSM8KDataset()
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    
    train_config = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_r * 2,
    }
    
    # 训练循环（与 verl 相同的结构）
    current_model = args.model_path
    
    for epoch in range(args.total_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.total_epochs}")
        print(f"{'='*80}")
        
        # 加载数据
        train_data = dataset.load("train", max_samples=args.max_samples)
        rl_prompts = [dataset.format_for_rl(item) for item in train_data]
        
        print(f"📊 {len(rl_prompts)} 个问题 × {args.samples_per_prompt} 次采样")
        print(f"   = {len(rl_prompts) * args.samples_per_prompt} 条轨迹")
        
        # Rollout
        print(f"\n🎲 Rollout")
        with RolloutManager(current_model, agent, tokenizer) as rollout_manager:
            trajectories = rollout_manager.generate_trajectories(
                rl_prompts,
                samples_per_prompt=args.samples_per_prompt,
                max_new_tokens=args.max_new_tokens  # 控制生成长度
            )
        
        avg_reward = sum(t["reward"] for t in trajectories) / len(trajectories)
        print(f"✅ {len(trajectories)} 条轨迹，平均 reward: {avg_reward:.3f}")
        
        # Training（每批轨迹只训练 1 遍）
        print(f"\n🏋️  Training")
        trainer = RLTrainer(train_config)
        trainer.load_model(current_model, use_lora=args.use_lora)
        trainer.train(trajectories, num_epochs=1)  # 固定为 1
        
        # 保存
        epoch_output_dir = f"{args.output_dir}/epoch_{epoch + 1}"
        trainer.save_model(epoch_output_dir)
        trainer.unload_model()
        
        current_model = epoch_output_dir
        print(f"✅ Checkpoint: {epoch_output_dir}")
    
    print(f"\n{'='*80}")
    print("🎉 RL 训练完成!")
    print(f"模型: {current_model}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
