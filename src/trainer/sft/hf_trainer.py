"""
SFT 训练器 (HuggingFace)
使用 transformers + PEFT (LoRA) + bitsandbytes (4-bit)
"""
import torch
from typing import List, Dict, Any
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class HFSFTTrainer:
    """
    基于 HuggingFace 的 SFT 训练器
    
    特性:
    1. 4-bit 量化
    2. LoRA 适配器
    3. 标准的监督微调
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.optimizer = None
    
    def load_model(self, model_path: str, use_lora: bool = True):
        """加载模型"""
        print(f"📦 加载模型: {model_path}")
        
        # 4-bit 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        if use_lora:
            self.model.gradient_checkpointing_enable()
            self.model = prepare_model_for_kbit_training(self.model)
            
            lora_config = LoraConfig(
                r=self.config.get("lora_r", 8),
                lora_alpha=self.config.get("lora_alpha", 16),
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=self.config.get("lora_dropout", 0.1),
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            print(f"   LoRA 参数: {self.model.print_trainable_parameters()}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ 模型加载完成")
    
    def train(
        self, 
        dataset: List[Dict[str, Any]], 
        num_epochs: int = 3
    ):
        """
        SFT 训练
        
        Args:
            dataset: 数据集，每项包含:
                - messages: List[Dict] (OpenAI 格式)
            num_epochs: 训练轮数
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        print(f"\n🎯 开始 SFT 训练...")
        print(f"   数据集大小: {len(dataset)}")
        print(f"   训练轮数: {num_epochs}")
        
        if self.optimizer is None:
            # Use 8-bit AdamW to save memory
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=self.config.get("learning_rate", 5e-5)
            )
        
        batch_size = self.config.get("batch_size", 1)
        grad_accum_steps = self.config.get("gradient_accumulation_steps", 4)
        
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                
                # Tokenize
                batch_data = self._prepare_batch(batch)
                
                # 前向传播
                loss = self._compute_loss(batch_data)
                
                # 梯度累积
                loss = loss / grad_accum_steps
                loss.backward()
                
                # 更新参数
                if (i // batch_size + 1) % grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()  # 清理显存避免碎片化
                
                epoch_loss += loss.item() * grad_accum_steps
                num_batches += 1
                
                if num_batches % 10 == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"  Batch {num_batches}, Loss: {avg_loss:.4f}")
            
            avg_epoch_loss = epoch_loss / num_batches
            print(f"  Epoch {epoch + 1} 平均 Loss: {avg_epoch_loss:.4f}")
    
    def _prepare_batch(self, batch: List[Dict]) -> Dict:
        """准备批次数据"""
        texts = []
        for item in batch:
            text = self.tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.get("max_length", 1024),  # 减少默认长度以节省显存
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
    
    def _compute_loss(self, batch: Dict) -> torch.Tensor:
        """计算损失"""
        device = next(self.model.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        outputs = self.model(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
            labels=input_ids[:, 1:],
            use_cache=False
        )
        
        return outputs.loss
    
    def save_model(self, output_dir: str):
        """保存模型"""
        print(f"💾 保存模型到: {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("✅ 模型保存完成")
    
    def unload_model(self):
        """卸载模型"""
        print("🗑️  卸载模型...")
        del self.model
        self.model = None
        torch.cuda.empty_cache()
        print("✅ 显存已释放")
