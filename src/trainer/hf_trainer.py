"""
RL 训练器 (HuggingFace)
使用 transformers + PEFT + 应用 loss-mask
"""
import torch
import torch.nn as nn
import logging
from typing import List, Dict, Any
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)


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
        
        # 自动选择设备
        if torch.cuda.is_available():
            device = "cuda"
            compute_dtype = torch.float16
            load_in_4bit = True
        elif torch.backends.mps.is_available():
            device = "mps"
            compute_dtype = torch.float16
            load_in_4bit = False # bitsandbytes 不支持 MPS
        else:
            device = "cpu"
            compute_dtype = torch.float32
            load_in_4bit = False

        # 加载模型
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=compute_dtype,
                device_map=None,
                trust_remote_code=True
            ).to(device)
        
        if use_lora:
            if load_in_4bit:
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
            if torch.cuda.is_available():
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    self.model.parameters(),
                    lr=self.config.get("learning_rate", 1e-5)
                )
            else:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config.get("learning_rate", 1e-5)
                )
        
        batch_size = self.config.get("batch_size", 1)
        grad_accum_steps = self.config.get("gradient_accumulation_steps", 4)
        
        if len(trajectories) == 0:
            print("⚠️ 没有轨迹可训练，跳过训练")
            return

        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(trajectories), batch_size):
                batch = trajectories[i:i + batch_size]
                if len(batch) == 0:
                    continue
                    
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
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                epoch_loss += loss.item() * grad_accum_steps
                num_batches += 1
                
                if num_batches % 10 == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"  Batch {num_batches}, Loss: {avg_loss:.4f}")
            
            if num_batches > 0:
                avg_epoch_loss = epoch_loss / num_batches
                print(f"  Epoch {epoch + 1} 平均 Loss: {avg_epoch_loss:.4f}")
            else:
                print(f"  Epoch {epoch + 1} 没有有效批次")
    
    def _prepare_batch(self, batch: List[Dict]) -> tuple:
        """准备 RL 批次数据"""
        # 限制最大长度以节省显存
        max_len = min(1024, max(len(traj["tokens"]) for traj in batch))
        
        batch_tokens = []
        batch_masks = []
        batch_advantages = [] # 改为使用 advantage
        
        for traj in batch:
            tokens = traj["tokens"]
            mask = traj["loss_mask"]
            advantage = traj.get("advantage", 0.0) # 使用 pre-computed advantage
            
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
            batch_advantages.append(advantage)
        
        # 转为 tensor
        tokens_tensor = torch.tensor(batch_tokens, dtype=torch.long)
        masks_tensor = torch.tensor(batch_masks, dtype=torch.float32)
        advantages_tensor = torch.tensor(batch_advantages, dtype=torch.float32)
        
        return tokens_tensor, masks_tensor, advantages_tensor

    def _compute_loss_with_mask(
        self,
        tokens: torch.Tensor,
        loss_masks: torch.Tensor,
        advantages: torch.Tensor, # 改名为 advantages
        trajectories: List[Dict]
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
        advantages = advantages.to(device)
        
        # 前向传播
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

        # Guard against NaN per_sample_loss
        if torch.isnan(per_sample_loss).any():
            logger.warning("[Trainer] NaN detected in per_sample_loss, replacing with 0")
            per_sample_loss = torch.where(torch.isnan(per_sample_loss), torch.zeros_like(per_sample_loss), per_sample_loss)

        # Advantage-weighted
        weighted_loss = (per_sample_loss * advantages).mean()
        
        return weighted_loss
    
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

# Alias for compatibility
RLTrainer = HFRLTrainer
