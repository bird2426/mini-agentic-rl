"""
GSM8K Agent - 数学问题求解
内置 calculator 和 code_executor 工具
"""
import re
from typing import Dict, Any, List
from .base import BaseAgent


class GSM8KAgent(BaseAgent):
    """GSM8K 数学问题求解 Agent（内置工具）"""
    
    def __init__(self):
        # 不再需要传入 tools
        super().__init__(tools=[])
        self.max_turns = 3
        
        print("🧮 GSM8K Agent 初始化")
        print("  内置工具: calculator, code_executor")
    
    def _calculator(self, expression: str) -> str:
        """计算器工具"""
        try:
            result = eval(expression, {"__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"计算错误: {e}"
    
    def _code_executor(self, code: str) -> str:
        """代码执行工具"""
        try:
            local_vars = {}
            exec(code, {"__builtins__": {}}, local_vars)
            return str(local_vars.get("result", "No result"))
        except Exception as e:
            return f"执行错误: {e}"
    
    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """执行工具（覆盖基类方法）"""
        if tool_name == "calculator":
            return self._calculator(args.get("expression", ""))
        elif tool_name == "code_executor":
            return self._code_executor(args.get("code", ""))
        else:
            return f"未知工具: {tool_name}"
    
    def process_turn(
        self,
        prompt: str,
        completion: str,
        conversation: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """处理一轮对话"""
        # 检查工具调用
        tool_call = self.parse_tool_call(completion)
        
        if tool_call:
            # 执行工具
            result = self.execute_tool(tool_call["tool"], tool_call["args"])
            
            return {
                "done": False,
                "tool_result": result,
                "tool_call": tool_call,
            }
        
        # 没有工具调用，检查是否有答案
        if "####" in completion:
            return {
                "done": True,
                "final_answer": completion,
            }
        
        # 既没工具也没答案
        return {
            "done": True,
            "final_answer": completion,
        }
    
    def should_continue(self, conversation: List[Dict[str, Any]]) -> bool:
        """判断是否继续"""
        if conversation and conversation[-1].get("role") == "tool":
            return True
        return super().should_continue(conversation)
