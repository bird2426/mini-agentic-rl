import sys
import os
from pathlib import Path

# 优化显存分配，避免碎片化
# PyTorch 2.0+ 推荐使用 PYTORCH_ALLOC_CONF (原 PYTORCH_CUDA_ALLOC_CONF)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from src.datasets import GSM8KDataset
from src.trainer import SFTTrainer


def main():
    parser = argparse.ArgumentParser(description="SFT 训练")
    
    # 模型
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-0.5B",
                       help="基础模型路径")
    parser.add_argument("--output_dir", type=str, default="./outputs/Qwen2.5-0.5B/sft",
                       help="输出目录")
    
    # 数据
    parser.add_argument("--dataset", type=str, default="gsm8k",
                       choices=["gsm8k"])
    
    # 训练
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="梯度累积")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="学习率")
    
    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=8)
    
    # 调试选项
    parser.add_argument("--max_samples", type=int, default=None,
                       help="限制样本数用于快速测试 (默认 None=全部)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SFT 训练")
    print("=" * 80)
    print(f"模型: {args.model_path}")
    print(f"数据集: {args.dataset} ({'全部' if args.max_samples is None else f'{args.max_samples} 样本'})")
    print(f"训练: {args.num_epochs} epochs, batch_size={args.batch_size}")
    print("=" * 80)
    
    # 加载数据集
    print("\n📥 加载数据集...")
    if args.dataset == "gsm8k":
        dataset = GSM8KDataset()
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    
    train_data = dataset.load("train", max_samples=args.max_samples)
    sft_data = [dataset.format_for_sft(item) for item in train_data]
    
    print(f"✅ 加载了 {len(sft_data)} 个样本")
    print(f"   每个 epoch: {len(sft_data)} / {args.batch_size} / {args.gradient_accumulation_steps}")
    print(f"              = {len(sft_data) // (args.batch_size * args.gradient_accumulation_steps)} 次参数更新")
    
    # 训练
    train_config = {
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_r * 2,
    }
    
    trainer = SFTTrainer(train_config)
    trainer.load_model(args.model_path, use_lora=args.use_lora)
    trainer.train(sft_data, num_epochs=args.num_epochs)
    trainer.save_model(args.output_dir)
    
    print(f"\n{'='*80}")
    print("🎉 训练完成!")
    print(f"模型: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
