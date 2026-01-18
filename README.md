# Mini Agentic RL

最小化实现的 Agentic RL 框架，用于训练具有工具调用能力的语言模型。

## 特性

- ✅ **简洁架构**: Agent 内置工具，无过度抽象
- ✅ **HuggingFace 生态**: 基于 transformers + PEFT
- ✅ **GRPO 训练**: Group Relative Policy Optimization
- ✅ **低显存友好**: 支持 4GB 显存训练（4-bit + LoRA）
- ✅ **模块化设计**: Agent、Dataset、Rollout、Trainer 解耦

## 快速开始

### 安装

```bash
conda create -n mini-agentic-rl python=3.10
conda activate mini-agentic-rl
pip install -r requirements.txt
```

### 训练流程

**1. SFT 训练**
```bash
python scripts/train_sft.py \
    --model_path Qwen/Qwen2.5-0.5B \
    --dataset gsm8k \
    --num_epochs 3 \
    --output_dir ./outputs/Qwen2.5-0.5B/sft
```

**2. RL 训练**
```bash
python scripts/train_gsm8k_agent.py \
    --model_path ./outputs/Qwen2.5-0.5B/sft \
    --dataset gsm8k \
    --samples_per_prompt 4 \
    --max_new_tokens 256 \
    --total_epochs 3 \
    --output_dir ./outputs/Qwen2.5-0.5B/rl
```

**3. 交互测试**
```bash
python scripts/test_interactive.py \
    --model_path ./outputs/Qwen2.5-0.5B/rl/epoch_3
```

## 架构

```
src/
├── agents/          # Agent 实现（内置工具）
├── datasets/        # 数据集加载
├── rollout/         # Rollout 管理（生成轨迹）
├── trainer/         # 训练器（SFT + RL）
└── utils/           # 工具函数
```

## 参数说明

### SFT 训练
- `--model_path`: 基础模型路径
- `--num_epochs`: 训练轮数
- `--batch_size`: 批次大小（显存限制）
- `--gradient_accumulation_steps`: 梯度累积

### RL 训练
- `--model_path`: SFT 模型路径
- `--samples_per_prompt`: GRPO 采样次数（每个问题采样几次）
- `--max_new_tokens`: 生成长度限制（控制推理速度）
- `--total_epochs`: RL 迭代次数（每轮: Rollout → Training）
- `--batch_size`: 批次大小
- `--gradient_accumulation_steps`: 梯度累积

### 交互测试
- `--model_path`: 模型路径
- `--max_new_tokens`: 最大生成长度
- `--temperature`: 温度参数

## 示例

### 快速测试（小数据集）
```bash
# SFT
python scripts/train_sft.py \
    --max_samples 10 \
    --num_epochs 1

# RL
python scripts/train_gsm8k_agent.py \
    --max_samples 5 \
    --samples_per_prompt 2 \
    --max_new_tokens 128 \
    --total_epochs 1

# 测试
python scripts/test_interactive.py \
    --model_path ./outputs/Qwen2.5-0.5B/rl/epoch_1
```

### 正式训练（全数据集）
```bash
# SFT
python scripts/train_sft.py \
    --num_epochs 3

# RL
python scripts/train_gsm8k_agent.py \
    --samples_per_prompt 4 \
    --max_new_tokens 256 \
    --total_epochs 5
```

## License

MIT
