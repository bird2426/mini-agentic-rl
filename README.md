# Mini Agentic RL

Minimal Agentic RL framework for Mac (MPS) training. Follows agent-lighting architecture.

## Quick Start

```bash
# Install
conda create -n mini-agentic-rl python=3.10
conda activate mini-agentic-rl
pip install -r requirements.txt

# SFT Training
python scripts/train.py --config scripts/configs/sft_gsm8k.yaml

# GRPO Training
python scripts/train.py --config scripts/configs/grpo_gsm8k.yaml
```

## Project Structure

```
scripts/
├── train.py              # Unified training script (SFT + GRPO)
└── configs/
    ├── sft_gsm8k.yaml   # SFT config
    └── grpo_gsm8k.yaml  # GRPO config

src/
├── agents/gsm8k_lit.py   # GSM8K Agent (LitAgent pattern)
├── algorithm/grpo.py      # GRPO algorithm
├── core/                  # Span, Triplet, Rollout types
├── datasets/gsm8k.py     # GSM8K dataset loader
├── rollout/hf_server.py   # HuggingFace inference server
├── runner/agent_runner.py # Async rollout worker
├── store/memory.py        # InMemory state store
└── trainer/hf_trainer.py  # HuggingFace trainer with LoRA
```

## Features

- MPS/CPU training support
- GSM8K math reasoning dataset
- GRPO RL training
- YAML-based configuration
- Async rollout processing

## License

MIT
