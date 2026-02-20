"""
Math Agent Demo - 展示 Agent-Lightning 的正确使用方式

这个示例展示如何:
1. 定义自定义 Agent (继承 LitAgent)
2. 使用 Hooks 计算 Reward
3. 使用 Trainer 编排训练
"""
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from mini_agent_lighting.algorithm.mini_grpo import MiniGRPO
from mini_agent_lighting.litagent import LitAgent
from mini_agent_lighting.store.memory import InMemoryLightningStore
from mini_agent_lighting.trainer import Trainer
from mini_agent_lighting.tracer.dummy import DummyTracer
from mini_agent_lighting.types import Hook, NamedResources, Rollout, Task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MathTaskInput(BaseModel):
    question: str
    answer: str


class MathAgent(LitAgent[MathTaskInput]):
    """数学问题求解 Agent"""
    
    def __init__(self, llm_client=None, prompt_template: str = None, **kwargs):
        super().__init__(**kwargs)
        self.llm_client = llm_client
        self.prompt_template = prompt_template or (
            "Solve the following math problem step by step.\n\n"
            "Problem: {question}\n\n"
            "Solution:"
        )
    
    def _build_prompt(self, task_input: MathTaskInput) -> str:
        return self.prompt_template.format(question=task_input.question)
    
    def rollout(
        self,
        task: Task,
        resources: NamedResources,
        rollout: Rollout,
    ) -> Optional[float]:
        task_input = task.input
        prompt = self._build_prompt(task_input)
        
        if self.llm_client:
            response = self.llm_client.chat(prompt)
        else:
            response = f"Mock response for: {task_input.question[:50]}"
        
        rollout.output = response
        
        reward = self._compute_reward(response, task_input.answer)
        rollout.metadata["reward"] = reward
        
        return reward
    
    def _compute_reward(self, response: str, ground_truth: str) -> float:
        pred = self._extract_answer(response)
        if pred is None:
            return 0.0
        
        try:
            pred_num = int(pred.replace(",", "").strip())
            truth_num = int(ground_truth.replace(",", "").strip())
            if pred_num == truth_num:
                return 1.0
            elif abs(pred_num - truth_num) <= 1:
                return 0.5
        except:
            pass
        
        return 0.0
    
    def _extract_answer(self, response: str) -> Optional[str]:
        response = response.strip()
        
        if "####" in response:
            for line in reversed(response.split("\n")):
                if "####" in line:
                    return line.split("####")[-1].strip()
        
        response = re.sub(r'```[\s\S]*?```', '', response)
        
        match = re.search(r"(?:final answer|answer is|=|:)\s*(\d+)", response, re.IGNORECASE)
        if match:
            return match.group(1)
        
        match = re.search(r"\d+\s*[-+\*/]\s*\d+\s*=\s*(\d+)", response)
        if match:
            return match.group(1)
        
        numbers = re.findall(r"\d+", response)
        if numbers:
            return numbers[-1]
        
        return None


class LoggingHook(Hook):
    """日志 Hook - 记录训练过程"""
    
    async def on_rollout_start(
        self,
        agent: LitAgent,
        runner: Any,
        rollout: Rollout,
    ) -> None:
        logger.info(f"Starting rollout for task: {rollout.input}")
    
    async def on_rollout_end(
        self,
        agent: LitAgent,
        runner: Any,
        rollout: Rollout,
        spans: Any,
    ) -> None:
        reward = rollout.metadata.get("reward", 0.0)
        logger.info(f"Rollout complete. Reward: {reward}")


def create_sample_dataset() -> List[MathTaskInput]:
    return [
        MathTaskInput(question="There are 5 birds on a tree. 2 fly away. How many birds are left?", answer="3"),
        MathTaskInput(question="Tom has 10 apples. He gives 3 to Jerry. How many apples does Tom have now?", answer="7"),
        MathTaskInput(question="Lisa has 8 candies. She buys 5 more. How many candies does she have in total?", answer="13"),
        MathTaskInput(question="There are 15 students in a class. 6 go on a trip. How many are left?", answer="9"),
    ]


def main():
    logger.info("=" * 60)
    logger.info("Math Agent Demo - Agent-Lightning 正确使用方式")
    logger.info("=" * 60)
    
    dataset = create_sample_dataset()
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    agent = MathAgent()
    logger.info("Created MathAgent")
    
    hooks = [LoggingHook()]
    logger.info(f"Created {len(hooks)} hooks")
    
    store = InMemoryLightningStore()
    logger.info("Created InMemoryLightningStore")
    
    tracer = DummyTracer()
    logger.info("Created DummyTracer")
    
    logger.info("\n注意: 完整训练需要 MiniGRPO 算法 + Trainer 集成")
    logger.info("当前示例展示 Agent + Hooks 的正确使用方式")
    logger.info("\n要运行完整训练,需要:")
    logger.info("1. 配置 MiniGRPO 算法 (已实现)")
    logger.info("2. 启动 Mini-LLM-Server")
    logger.info("3. 使用 Trainer.fit() 编排")


if __name__ == "__main__":
    main()
