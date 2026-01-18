"""
Rollout 模块 - 轨迹生成
支持多种推理引擎: HF (默认), SGLang (可选)
"""
from .manager import RolloutManager
from .hf_engine import HFInferenceEngine

# SGLang 引擎 (可选，需要额外配置)
try:
    from .sglang_server import SGLangServer
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False

__all__ = [
    "RolloutManager",
    "HFInferenceEngine",
]

# 只在 SGLang 可用时导出
if SGLANG_AVAILABLE:
    __all__.append("SGLangServer")
