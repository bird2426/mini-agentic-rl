"""
RL 训练器 (HuggingFace)
使用 transformers + PEFT + 应用 loss-mask
"""
import torch
import torch.nn as nn
from typing import List, Dict, Any
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class HFRLTrainer:
    """
    基于 HuggingFace 的 RL 训练器
    
    特性:
    1. 4-bit 量化
    2. LoRA 适配器
    3. 应用 loss-mask
    4. Advantage-weighted loss
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
        trajectories: List[Dict[str, Any]], 
        num_epochs: int = 1
    ):
        """
        RL 训练 (应用 loss-mask)
        
        Args:
            trajectories: 轨迹列表，每项包含:
                - tokens: List[int]
                - loss_mask: List[int] (0/1)
                - reward: float
            num_epochs: 训练轮数
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        print(f"\n🎯 开始 RL 训练...")
        print(f"   轨迹数量: {len(trajectories)}")
        print(f"   训练轮数: {num_epochs}")
        
        if self.optimizer is None:
            # 使用 8-bit AdamW 节省显存
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=self.config.get("learning_rate", 1e-5)
            )
        
        batch_size = self.config.get("batch_size", 1)
        grad_accum_steps = self.config.get("gradient_accumulation_steps", 4)
        
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(trajectories), batch_size):
                batch = trajectories[i:i + batch_size]
                
                # 准备数据
                batch_tokens, batch_masks, batch_rewards = self._prepare_batch(batch)
                
                # 计算 loss (应用 mask + GRPO advantage)
                loss = self._compute_loss_with_mask(
                    batch_tokens,
                    batch_masks,
                    batch_rewards,
                    batch  # 传递完整的 batch 用于 group-relative advantage
                )
                
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
    
    def _prepare_batch(self, batch: List[Dict]) -> tuple:
        """准备 RL 批次数据"""
        # 限制最大长度以节省显存
        max_len = min(1024, max(len(traj["tokens"]) for traj in batch))
        
        batch_tokens = []
        batch_masks = []
        batch_rewards = []
        
        for traj in batch:
            tokens = traj["tokens"]
            mask = traj["loss_mask"]
            reward = traj["reward"]
            
            # Truncate
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
                mask = mask[:max_len]
            
            # Padding
            pad_len = max_len - len(tokens)
            tokens_padded = tokens + [self.tokenizer.pad_token_id] * pad_len
            mask_padded = mask + [0] * pad_len
            
            batch_tokens.append(tokens_padded)
            batch_masks.append(mask_padded)
            batch_rewards.append(reward)
        
        # 转为 tensor
        tokens_tensor = torch.tensor(batch_tokens, dtype=torch.long)
        masks_tensor = torch.tensor(batch_masks, dtype=torch.float32)
        rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32)
        
        return tokens_tensor, masks_tensor, rewards_tensor
    
    
    def _compute_loss_with_mask(
        self,
        tokens: torch.Tensor,
        loss_masks: torch.Tensor,
        rewards: torch.Tensor,
        trajectories: List[Dict]  # 新增
    ) -> torch.Tensor:
        """
        计算应用 loss-mask 的 RL 损失
        
        核心逻辑:
        1. 计算 per-token loss
        2. 应用 loss-mask
        3. 计算每个样本的平均 loss
        4. Advantage-weighted
        """
        device = next(self.model.parameters()).device
        tokens = tokens.to(device)
        loss_masks = loss_masks.to(device)
        rewards = rewards.to(device)
        
        # 前向传播
        # tokens 已经通过 _prepare_batch 进行了 max_length 限制和 padding
        outputs = self.model(
            input_ids=tokens[:, :-1],
            labels=tokens[:, 1:],
            use_cache=False
        )
        
        # 计算 per-token loss
        logits = outputs.logits
        vocab_size = logits.size(-1)
        
        shift_labels = tokens[:, 1:].contiguous()
        shift_masks = loss_masks[:, 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_loss = loss_fct(
            logits.view(-1, vocab_size),
            shift_labels.view(-1)
        ).view_as(shift_labels)
        
        # 应用 mask
        masked_loss = token_loss * shift_masks
        
        # 每个样本的平均 loss
        per_sample_loss = (
            masked_loss.sum(dim=1) / 
            shift_masks.sum(dim=1).clamp_min(1)
        )
        
        # Advantage-weighted
        advantages = self._compute_advantages(rewards)
        weighted_loss = (per_sample_loss * advantages).mean()
        
        return weighted_loss
    
    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """计算优势函数"""
        if len(rewards) == 1:
            return torch.ones_like(rewards)
        
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return advantages
    
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
