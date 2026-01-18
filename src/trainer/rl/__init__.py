"""
RL 训练器模块
"""
from .hf_trainer import HFRLTrainer

# 默认使用 HF
RLTrainer = HFRLTrainer

__all__ = [
    "HFRLTrainer",
    "RLTrainer",
]
