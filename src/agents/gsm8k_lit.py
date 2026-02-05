from __future__ import annotations
import re
import torch
import httpx
from typing import Any, Dict, List, Optional, Union
from src.core.agent import LitAgent
from src.core.types import Rollout, NamedResources, Span
from src.core.tracer import Tracer

class GSM8KLitAgent(LitAgent):
    def __init__(self, max_turns: int = 3, use_server: bool = False):
        super().__init__()
        self.max_turns = max_turns
        self.use_server = use_server
        self._model = None
        self._tokenizer = None
        self._device = None

    def _setup_model(self, resources: NamedResources):
        """Setup model from resources"""
        if self._model is None:
            self._model = resources.resources.get("model")
            self._tokenizer = resources.resources.get("tokenizer")
            if self._model is not None:
                self._device = next(self._model.parameters()).device
                print(f"[GSM8KAgent] Model device: {self._device}")
                print(f"[GSM8KAgent] Model type: {type(self._model)}")

    def _calculator(self, expression: str) -> str:
        try:
            result = eval(expression, {"__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    def _generate(self, messages: List[Dict], resources: NamedResources, max_tokens: int = 64) -> str:
        """Generate response"""
        if self.use_server:
            llm_config = resources.resources.get("llm", {})
            endpoint = llm_config.get("endpoint", "http://localhost:8000/v1/chat/completions")

            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    endpoint,
                    json={"messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        else:
            # Direct model call
            self._setup_model(resources)
            if self._model is None:
                return "Error: Model not available"

            if self._tokenizer is None:
                return "Error: Tokenizer not available"

            encoded = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            # Handle BatchEncoding object
            if hasattr(encoded, 'input_ids'):
                input_ids = encoded.input_ids.to(self._device)
            else:
                input_ids = encoded.to(self._device)

            if input_ids is None:
                return "Error: Failed to tokenize input"

            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id
                )

            completion_ids = outputs[0][input_ids.shape[1]:]
            return self._tokenizer.decode(completion_ids, skip_special_tokens=True)

    async def rollout_async(self, task_input: Any, resources: NamedResources, rollout: Rollout) -> List[Span]:
        tracer = Tracer(rollout.rollout_id, rollout.attempt.attempt_id)

        # Handle both string and dict inputs
        if isinstance(task_input, dict):
            prompt = task_input.get("prompt", "")
            rollout.metadata["ground_truth"] = task_input.get("ground_truth", "")
        else:
            prompt = task_input
        
        messages = [{"role": "user", "content": prompt}]
        
        # Generate responses
        for turn in range(self.max_turns):
            with tracer.span("generation", attributes={"messages": list(messages)}) as attrs:
                completion_text = self._generate(messages, resources, max_tokens=64)
                attrs["output"] = completion_text
                
            messages.append({"role": "assistant", "content": completion_text})

            tool_match = re.search(r'<tool>(\w+)</tool>', completion_text)
            args_match = re.search(r'<args>(.*?)</args>', completion_text, re.DOTALL)
            
            if tool_match:
                tool_name = tool_match.group(1)
                args_str = args_match.group(1) if args_match else ""
                
                with tracer.span("tool_call", attributes={"tool": tool_name, "args": args_str}) as attrs:
                    if tool_name == "calculator":
                        result = self._calculator(args_str)
                    else:
                        result = f"Unknown tool: {tool_name}"
                    attrs["output"] = result
                
                messages.append({"role": "tool", "content": result})
            else:
                break

        final_answer = messages[-1]["content"]
        ground_truth = rollout.metadata.get("ground_truth", "")
        ground_truth_str = str(ground_truth).strip()

        answer_match = re.search(r'The answer is (\d+)', final_answer)
        if answer_match:
            final_num = answer_match.group(1)
            gt_nums = re.findall(r'\d+', str(ground_truth))
            reward = 1.0 if gt_nums and final_num == gt_nums[-1] else 0.0
        else:
            reward = 1.0 if ground_truth_str in final_answer else 0.0
        
        with tracer.span("reward", attributes={"value": reward}):
            pass
        
        return tracer.get_spans()
