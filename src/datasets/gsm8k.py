"""GSM8K Dataset - 数学问题数据集

继承 BaseDataset，实现 GSM8K 特定的加载和格式化逻辑
"""

from typing import List, Dict, Any
from .base import BaseDataset


class GSM8KDataset(BaseDataset):
    """GSM8K 数学问题数据集"""
    
    def __init__(self):
        print("📚 GSM8K Dataset")
    
    def load(self, split: str = "train", max_samples: int = None) -> List[Dict[str, Any]]:
        """加载 GSM8K 数据"""
        from datasets import load_dataset
        
        print(f"📥 加载 GSM8K ({split})...")
        
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        data = []
        for item in dataset:
            data.append({
                "question": item["question"],
                "answer": item["answer"],
            })
        
        print(f"  加载 {len(data)} 个样本")
        return data
    
    def format_for_sft(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化为 SFT 训练格式 (OpenAI messages 格式)
        
        Returns:
            {"messages": [{"role": "user", "content": "..."}, ...]}
        """
        question = item['question']
        answer = item['answer']
        
        # 构建 messages (OpenAI 格式)
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"Let's solve this step by step:\n\n{answer}"}
        ]
        
        return {"messages": messages}
    
    def format_for_rl(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """RL 格式: prompt + ground_truth"""
        prompt = f"""Question: {item['question']}

Let's solve this step by step:

"""
        
        # 提取最终答案
        answer = item['answer']
        if '####' in answer:
            ground_truth = answer.split('####')[1].strip()
        else:
            ground_truth = answer
        
        return {
            "prompt": prompt,
            "ground_truth": ground_truth,
        }


if __name__ == "__main__":
    # 测试
    dataset = GSM8KDataset()
    
    # 测试加载
    data = dataset.load(split="train", max_samples=2)
    print(f"\n原始数据: {data[0]}")
    
    # 测试 SFT 格式
    sft = dataset.format_for_sft(data[0])
    print(f"\nSFT 格式:\n{sft['text'][:200]}...")
    
    # 测试 RL 格式
    rl = dataset.format_for_rl(data[0])
    print(f"\nRL 格式:\nPrompt: {rl['prompt'][:100]}...")
    print(f"GT: {rl['ground_truth']}")
