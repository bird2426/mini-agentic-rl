"""
训练器模块 - 支持 SFT 和 RL
"""
from .sft import HFSFTTrainer, SFTTrainer
from .rl import HFRLTrainer, RLTrainer

__all__ = [
    # SFT
    "HFSFTTrainer",
    "SFTTrainer",
    
    # RL
    "HFRLTrainer",
    "RLTrainer",
]
