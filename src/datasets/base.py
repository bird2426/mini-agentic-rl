"""Dataset 基类 - 定义数据集接口"""

from typing import List, Dict, Any


class BaseDataset:
    """数据集基类
    
    子类需要实现:
    - load(): 加载数据
    - format_for_sft(): SFT 格式化
    - format_for_rl(): RL 格式化
    """
    
    def load(self, split: str = "train", max_samples: int = None) -> List[Dict[str, Any]]:
        """加载数据集
        
        Args:
            split: 数据集分割 (train/test)
            max_samples: 最大样本数
            
        Returns:
            原始数据列表
        """
        raise NotImplementedError
    
    def format_for_sft(self, item: Dict[str, Any]) -> Dict[str, str]:
        """格式化为 SFT 训练格式
        
        Args:
            item: 原始数据项
            
        Returns:
            {"text": "prompt + answer"}
        """
        raise NotImplementedError
    
    def format_for_rl(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """格式化为 RL 训练格式
        
        Args:
            item: 原始数据项
            
        Returns:
            {
                "prompt": "question",
                "ground_truth": "answer"
            }
        """
        raise NotImplementedError
    
    def load_for_sft(self, split: str = "train", max_samples: int = None) -> List[Dict[str, str]]:
        """加载并格式化为 SFT 格式"""
        data = self.load(split, max_samples)
        return [self.format_for_sft(item) for item in data]
    
    def load_for_rl(self, split: str = "train", max_samples: int = None) -> List[Dict[str, Any]]:
        """加载并格式化为 RL 格式"""
        data = self.load(split, max_samples)
        return [self.format_for_rl(item) for item in data]


if __name__ == "__main__":
    print("BaseDataset 基类定义")
