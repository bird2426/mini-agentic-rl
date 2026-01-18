"""
HuggingFace 推理引擎
使用 transformers 进行推理，适合小显存环境
"""
import torch
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFInferenceEngine:
    """
    基于 HuggingFace 的推理引擎
    
    特性:
    - 直接生成 token IDs
    - 支持 fp16 节省显存
    - 适合 4GB 显存
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
    
    def load(self):
        """加载模型"""
        print(f"📦 加载推理模型: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # 设置 pad_token（修复 attention_mask 警告）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            dtype=torch.float16,  # 修复: torch_dtype → dtype
            trust_remote_code=True
        )
        self.model.eval()
        
        print("✅ 推理模型加载完成")
    
    def generate_tokens(
        self,
        input_ids: List[int],
        max_new_tokens: int = 256,  # 降低默认值，加快推理
        temperature: float = 0.7,
    ) -> List[int]:
        """
        生成并返回完整的 token 序列
        
        Args:
            input_ids: 输入的 token IDs
            max_new_tokens: 最大生成长度（降低可加快推理）
            temperature: 温度参数
        
        Returns:
            完整的 token 序列 (input + generated)
        """
        device = next(self.model.parameters()).device
        
        # 转为 tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        # 创建 attention_mask（修复警告）
        attention_mask = torch.ones_like(input_tensor)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                input_tensor,
                attention_mask=attention_mask,  # 添加 attention_mask
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 返回完整序列
        return outputs[0].tolist()
    
    def unload(self):
        """卸载模型释放显存"""
        print("🗑️  卸载推理模型...")
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()
        print("✅ 显存已释放")
    
    def __enter__(self):
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
