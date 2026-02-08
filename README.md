# Mini Agentic RL

Agent-Lightning-inspired mini framework for GSM8K RL training on a single machine.

## Features

- decoupled pipeline: rollout collection, rewarding, optimization, evaluation
- GRPO-style grouped sampling with advantage normalization
- HF + PEFT + 4-bit quantization defaults for limited VRAM
- standalone train/eval/interactive scripts

## 快速开始

### Install

```bash
conda env create -f environment.yml
conda activate mini-agentic-rl
```

### Training Flow

**1) Optional SFT warm start**
```bash
python scripts/preprocess/train_sft.py \
    --model_path Qwen/Qwen2.5-0.5B \
    --num_epochs 2 \
    --output_dir ./outputs/Qwen2.5-0.5B/sft
```

**2) RL training**
```bash
python scripts/train_gsm8k_agent.py \
    --model_path ./outputs/Qwen2.5-0.5B/sft \
    --samples_per_prompt 4 \
    --total_epochs 2 \
    --max_train_samples 512 \
    --max_eval_samples 128 \
    --output_dir ./outputs/mini_gsm8k_rl
```

**3) Evaluate**
```bash
python scripts/eval_gsm8k.py \
    --model_path ./outputs/mini_gsm8k_rl/best \
    --split test \
    --max_samples 200
```

**4) Interactive test**
```bash
python scripts/test_interactive.py \
    --model_path ./outputs/mini_gsm8k_rl/best
```

## Architecture

```
src/
├── mini_rl/
│   ├── store.py       # rollout queue + attempts + spans + resources
│   ├── agent.py       # rollout execution logic
│   ├── runner.py      # worker loop
│   ├── adapter.py     # spans -> triplets
│   ├── algorithm.py   # triplets -> optimization
│   ├── trainer.py     # orchestration
│   ├── backend/       # HF backends (RL/SFT)
│   ├── data/          # gsm8k loading
│   ├── model/         # policy wrapper
│   ├── rl/            # grpo math
│   ├── reward.py      # reward + answer extraction
│   └── pipeline.py    # prompt + generation path

scripts/
├── preprocess/train_sft.py
├── train_gsm8k_agent.py
├── eval_gsm8k.py
└── test_interactive.py
```

See `docs/mini-agent-lightning-architecture.md` for detailed design.

## License

MIT
