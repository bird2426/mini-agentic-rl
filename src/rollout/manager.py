"""
Rollout Manager - 负责轨迹生成
使用可插拔的推理引擎架构
"""
from typing import List, Dict, Any
from transformers import AutoTokenizer


class RolloutManager:
    """
    Rollout 管理器 - 生成训练轨迹
    
    核心设计:
    1. Rollout 时直接记录 token IDs (不是文本)
    2. 根据来源标记 loss-mask:
       - User → 0 (不训练)
       - Assistant → 1 (训练)
       - Tool → 0 (不训练)
    3. 支持可插拔的推理引擎 (HF / SGLang / ...)
    """
    
    def __init__(
        self,
        model_path: str,
        agent,  # BaseAgent
        tokenizer: AutoTokenizer,
        inference_engine=None  # 可选的推理引擎
    ):
        self.model_path = model_path
        self.agent = agent
        self.tokenizer = tokenizer
        
        # 推理引擎 (默认使用 HF)
        if inference_engine is None:
            from .hf_engine import HFInferenceEngine
            self.engine = HFInferenceEngine(model_path)
        else:
            self.engine = inference_engine
        
        self.started = False
    
    def start(self):
        """启动 Rollout 环境"""
        if not self.started:
            self.engine.load()
            self.started = True
    
    def _tokenize_user_message(self, content: str) -> List[int]:
        """Tokenize user 消息"""
        message = {"role": "user", "content": content}
        tokens = self.tokenizer.apply_chat_template(
            [message],
            tokenize=True,
            add_generation_prompt=True
        )
        return tokens
    
    def _tokenize_tool_output(self, content: str) -> List[int]:
        """Tokenize tool 输出"""
        # 简单实现: 直接 encode
        tokens = self.tokenizer.encode(
            f"\nTool Result: {content}\n\nAssistant: ",
            add_special_tokens=False
        )
        return tokens
    
    
    def generate_trajectory(
        self,
        prompt: str,
        ground_truth: Any,
        max_turns: int = 10,
        max_new_tokens: int = 256  # 降低默认值，加快推理
    ) -> Dict[str, Any]:
        """
        生成一条轨迹
        
        Returns:
            {
                "tokens": List[int],
                "loss_mask": List[int],
                "reward": float,
                "response_length": int,
                "messages": List[Dict],
            }
        """
        # 初始化
        all_token_ids = []
        all_loss_mask = []
        messages = []
        
        # Turn 1: User prompt
        user_tokens = self._tokenize_user_message(prompt)
        all_token_ids.extend(user_tokens)
        all_loss_mask.extend([0] * len(user_tokens))
        messages.append({"role": "user", "content": prompt})
        
        # 多轮生成
        for turn in range(max_turns):
            prev_length = len(all_token_ids)
            
            # 推理引擎生成
            full_tokens = self.engine.generate_tokens(
                all_token_ids,
                max_new_tokens=max_new_tokens
            )
            
            # 计算增量 (Assistant 的输出)
            assistant_tokens = full_tokens[prev_length:]
            
            # 更新 token 序列和 mask
            all_token_ids.extend(assistant_tokens)
            all_loss_mask.extend([1] * len(assistant_tokens))  # 训练!
            
            # 解码用于 Agent 处理
            assistant_text = self.tokenizer.decode(
                assistant_tokens,
                skip_special_tokens=True
            )
            messages.append({"role": "assistant", "content": assistant_text})
            
            # Agent 处理
            result = self.agent.process_turn(prompt, assistant_text, messages)
            
            if result.get("done", False):
                break
            
            # Tool 输出 (如果有)
            if "tool_result" in result:
                tool_text = result["tool_result"]
                tool_tokens = self._tokenize_tool_output(tool_text)
                
                all_token_ids.extend(tool_tokens)
                all_loss_mask.extend([0] * len(tool_tokens))  # 不训练!
                
                messages.append({"role": "tool", "content": tool_text})
        
        # 计算 reward
        reward = self._compute_reward(messages, ground_truth)
        
        return {
            "tokens": all_token_ids,
            "loss_mask": all_loss_mask,
            "reward": reward,
            "response_length": len(all_token_ids),
            "messages": messages,
        }
    
    def _compute_reward(
        self,
        messages: List[Dict],
        ground_truth: Any
    ) -> float:
        """计算 reward"""
        # 提取最后一个 assistant 的回复
        final_answer = None
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                final_answer = msg["content"]
                break
        
        if final_answer is None:
            return 0.0
        
        # 简单的匹配检查
        if str(ground_truth) in final_answer:
            return 1.0
        else:
            return 0.0
    
    
    def generate_trajectories(
        self,
        dataset: List[Dict],
        samples_per_prompt: int = 4,
        max_new_tokens: int = 256  # 新增参数
    ) -> List[Dict]:
        """
        批量生成轨迹 (GRPO 风格)
        
        Args:
            dataset: 数据集
            samples_per_prompt: 每个 prompt 采样多少条轨迹
            max_new_tokens: 每次生成的最大 token 数
        
        Returns:
            轨迹列表，包含 group_id 用于计算 group-relative advantage
        """
        trajectories = []
        
        for group_id, item in enumerate(dataset):
            print(f"\n📝 Prompt {group_id+1}/{len(dataset)}: {item['prompt'][:50]}...")
            
            # 对同一个 prompt 采样多次
            for sample_id in range(samples_per_prompt):
                print(f"  采样 {sample_id+1}/{samples_per_prompt}...", end=" ")
                
                traj = self.generate_trajectory(
                    item["prompt"],
                    item["ground_truth"],
                    max_new_tokens=max_new_tokens  # 传递参数
                )
                
                # 添加 group_id 用于计算 group-relative advantage
                traj["group_id"] = group_id
                traj["sample_id"] = sample_id
                
                trajectories.append(traj)
                
                print(f"Reward: {traj['reward']:.2f}, Length: {traj['response_length']}")
        
        return trajectories
    
    def shutdown(self):
        """关闭 Rollout 环境"""
        if self.started:
            self.engine.unload()
            self.started = False
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
