"""
奖励函数模块

用于 RL 训练的奖励计算
"""

import re
from typing import List


class AccuracyReward:
    """准确性奖励函数 - 答案正确给 1.0，错误给 0.0"""
    
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        计算奖励
        
        Args:
            completions: 模型生成的回答列表
            **kwargs: 必须包含 ground_truth (正确答案列表)
            
        Returns:
            奖励列表
        """
        ground_truths = kwargs.get("ground_truth", [])
        
        if len(completions) != len(ground_truths):
            raise ValueError(
                f"completions 和 ground_truth 长度不匹配: "
                f"{len(completions)} vs {len(ground_truths)}"
            )
        
        rewards = []
        for completion, gt in zip(completions, ground_truths):
            # 提取模型回答中的数字
            predicted = self._extract_answer(completion)
            expected = self._extract_answer(str(gt))
            
            # 比较答案
            reward = 1.0 if predicted == expected else 0.0
            rewards.append(reward)
        
        return rewards
    
    def _extract_answer(self, text: str) -> str:
        """从文本中提取最终答案"""
        # 尝试匹配 #### 后面的数字
        match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
        if match:
            return match.group(1).replace(",", "")
        
        # 尝试匹配最后一个数字
        numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
        if numbers:
            return numbers[-1].replace(",", "")
        
        return ""


def create_accuracy_reward():
    """创建准确性奖励函数"""
    return AccuracyReward()


class LengthPenaltyReward:
    """带长度惩罚的奖励函数 - 鼓励简洁的回答"""
    
    def __init__(self, base_reward_fn, penalty_weight: float = 0.001, max_length: int = 512):
        """
        Args:
            base_reward_fn: 基础奖励函数
            penalty_weight: 惩罚权重
            max_length: 最大长度
        """
        self.base_reward_fn = base_reward_fn
        self.penalty_weight = penalty_weight
        self.max_length = max_length
    
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """计算带长度惩罚的奖励"""
        base_rewards = self.base_reward_fn(completions, **kwargs)
        
        rewards = []
        for base_reward, completion in zip(base_rewards, completions):
            length_penalty = (len(completion) / self.max_length) * self.penalty_weight
            reward = base_reward - length_penalty
            rewards.append(reward)
        
        return rewards


def create_length_penalty_reward(base_reward_fn=None, penalty_weight: float = 0.001):
    """创建带长度惩罚的奖励函数"""
    if base_reward_fn is None:
        base_reward_fn = create_accuracy_reward()
    return LengthPenaltyReward(base_reward_fn, penalty_weight)


if __name__ == "__main__":
    # 测试奖励函数
    reward_fn = create_accuracy_reward()
    
    completions = [
        "Let's solve: 2+2=4\n#### 4",
        "The answer is 5",
    ]
    ground_truths = ["4", "4"]
    
    rewards = reward_fn(completions, ground_truth=ground_truths)
    print("奖励:", rewards)  # 应该是 [1.0, 0.0]
