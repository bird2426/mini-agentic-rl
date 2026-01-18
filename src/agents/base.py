"""Agent 基类 - 定义 Agent 接口

每个 Agent 实现自己的业务逻辑，而不是硬编码在框架中
"""

from typing import Dict, Any, List, Optional


class BaseAgent:
    """Agent 基类
    
    子类需要实现:
    - process_turn(): 处理一轮对话
    - should_continue(): 判断是否继续生成
    """
    
    def __init__(self, tools: Optional[List] = None):
        """
        初始化 Agent
        
        Args:
            tools: 可用工具列表
        """
        self.tools = {t.name: t for t in (tools or [])}
        self.max_turns = 5  # 默认最大轮次
    
    def process_turn(
        self,
        prompt: str,
        completion: str,
        conversation: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """处理一轮对话
        
        Args:
            prompt: 当前 prompt
            completion: 模型生成的回复
            conversation: 对话历史
            
        Returns:
            {
                "continue": bool,  # 是否继续生成
                "next_prompt": str,  # 下一轮的 prompt (如果 continue=True)
                "result": Any,  # 工具执行结果 (可选)
            }
        """
        raise NotImplementedError("子类必须实现 process_turn 方法")
    
    def should_continue(self, conversation: List[Dict[str, Any]]) -> bool:
        """判断是否应该继续生成
        
        Args:
            conversation: 对话历史
            
        Returns:
            是否继续
        """
        # 默认: 检查轮次
        assistant_turns = len([m for m in conversation if m.get("role") == "assistant"])
        return assistant_turns < self.max_turns
    
    def parse_tool_call(self, completion: str) -> Optional[Dict[str, Any]]:
        """解析工具调用 (子类可覆盖)
        
        默认支持简单格式: <tool>name</tool><args>{...}</args>
        """
        import re
        import json
        
        tool_match = re.search(r'<tool>(\w+)</tool>', completion)
        args_match = re.search(r'<args>(.*?)</args>', completion, re.DOTALL)
        
        if tool_match:
            tool_name = tool_match.group(1)
            args = {}
            if args_match:
                try:
                    args = json.loads(args_match.group(1))
                except:
                    pass
            
            return {"tool": tool_name, "args": args}
        
        return None
    
    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """执行工具"""
        if tool_name not in self.tools:
            return f"错误: 工具 '{tool_name}' 不存在"
        
        try:
            return self.tools[tool_name].execute(**args)
        except Exception as e:
            return f"工具执行错误: {str(e)}"


if __name__ == "__main__":
    print("BaseAgent 基类定义")
