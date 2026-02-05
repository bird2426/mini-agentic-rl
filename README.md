# Mini Agentic RL

Cross-platform Agentic RL framework. Works on Mac (MPS), GPU (CUDA), and CPU.

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

## Cross-Platform Support

Configure device in YAML:
```yaml
device: auto    # Auto-detect (mps/cuda/cpu)
device: mps      # Mac Apple Silicon
device: cuda    # NVIDIA GPU
device: cpu     # CPU fallback
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
├── datasets/gsm8k.py    # GSM8K dataset loader
├── rollout/hf_server.py   # HuggingFace inference server
├── runner/agent_runner.py # Async rollout worker
├── store/memory.py        # InMemory state store
└── trainer/hf_trainer.py  # HuggingFace trainer with LoRA
```

## Features

- **Cross-Platform**: Mac (MPS), GPU (CUDA), CPU
- **GRPO Training**: Group Relative Policy Optimization
- **Async Rollout**: Parallel trajectory collection
- **YAML Config**: Reproducible experiments
- **LoRA Support**: Memory-efficient fine-tuning

## License

MIT
