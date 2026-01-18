"""
SFT 训练器模块
"""
from .hf_trainer import HFSFTTrainer

# 默认使用 HF
SFTTrainer = HFSFTTrainer

__all__ = [
    "HFSFTTrainer",
    "SFTTrainer",
]
