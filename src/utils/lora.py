"""
LoRA 配置工具
"""

from peft import LoraConfig, get_peft_model


def setup_lora(
    model,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list = None
):
    """
    为模型配置 LoRA
    
    Args:
        model: 基础模型
        r: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        target_modules: 目标模块
        
    Returns:
        配置了 LoRA 的模型
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    return model
