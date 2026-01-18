"""
Mini Agentic RL - 简洁架构 v4.0

核心设计:
- Agent 内置工具
- 可插拔的推理引擎 (HF)
- 可插拔的训练后端 (HF)
"""

__version__ = "4.0.0"

# Agents
from .agents import BaseAgent, GSM8KAgent

# Datasets
from .datasets import BaseDataset, GSM8KDataset

# Rollout
from .rollout import RolloutManager, HFInferenceEngine

# Trainer
from .trainer import SFTTrainer, RLTrainer, HFSFTTrainer, HFRLTrainer

__all__ = [
    # Agents
    "BaseAgent",
    "GSM8KAgent",
    
    # Datasets
    "BaseDataset",
    "GSM8KDataset",
    
    # Rollout
    "RolloutManager",
    "HFInferenceEngine",
    
    # Trainer
    "SFTTrainer",
    "RLTrainer",
    "HFSFTTrainer",
    "HFRLTrainer",
]
