"""
SGLang 服务器管理 - 独立进程模式
支持返回 token IDs 以避免 tokenizer 不一致
优化配置适配 4GB 显存
"""
import time
import subprocess
import requests
import os
from typing import List, Dict, Any, Optional


class SGLangServer:
    """SGLang 服务器管理器 (独立进程模式)"""
    
    def __init__(
        self, 
        model_path: str,
        tp_size: int = 1,
        port: int = 30000,
        mem_fraction: float = 0.5  # 4GB GPU 需要更少
    ):
        self.model_path = model_path
        self.tp_size = tp_size
        self.port = port
        self.mem_fraction = mem_fraction
        self.process = None
        self.url = f"http://localhost:{port}"
    
    def start(self):
        """启动 SGLang 服务器 (4GB 显存优化)"""
        print(f"🚀 启动 SGLang 服务器: {self.model_path}")
        
        # 检查是否是 LoRA adapter
        if os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
            print("\n❌ 错误: 检测到 LoRA adapter")
            print("SGLang 需要完整的模型，不能直接加载 LoRA adapter")
            print("\n解决方案:")
            print("1. 使用原始基础模型: python scripts/interactive_test.py --model_path Qwen/Qwen2.5-0.5B")
            print("2. 或使用简单测试: python scripts/simple_test.py --model_path Qwen/Qwen2.5-0.5B")
            raise RuntimeError("不能直接加载 LoRA adapter")
        
        # 4GB 显存优化配置
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--port", str(self.port),
            
            # 核心显存配置
            "--mem-fraction-static", str(self.mem_fraction),  # 限制静态显存占用
            
            # 关闭显存消耗大的功能
            "--disable-cuda-graph",        # 关闭 CUDA graph (节省 ~512MB)
            "--disable-radix-cache",       # 关闭 radix cache
            
            # 限制并发和 token 数
            "--chunked-prefill-size", "512",    # 减小 chunked prefill (默认 2048)
            "--max-running-requests", "2",       # 限制并发请求数
            "--max-total-tokens", "4096",        # 限制总 token 数
            "--max-prefill-tokens", "2048",      # 限制 prefill tokens
            
            # CUDA graph 配置 (以防被启用)
            "--cuda-graph-max-bs", "1",          # 限制 batch size
            "--cuda-graph-bs", "1",              # 只 capture bs=1
        ]
        
        print(f"   🔧 4GB 显存优化配置:")
        print(f"      - mem_fraction: {self.mem_fraction}")
        print(f"      - ❌ CUDA graph (节省 ~512MB)")
        print(f"      - ❌ radix cache")
        print(f"      - chunked_prefill_size: 512")
        print(f"      - max_running_requests: 2")
        print(f"      - max_total_tokens: 4096")
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 等待服务器就绪
        self._wait_for_ready()
        print(f"✅ SGLang 服务器已就绪: {self.url}")
    
    def _wait_for_ready(self, timeout: int = 180):
        """等待服务器就绪 (增加超时时间，因为关闭了 CUDA graph)"""
        print("⏳ 等待服务器启动 (可能需要 1-2 分钟)...")
        start_time = time.time()
        last_log_time = start_time
        
        while time.time() - start_time < timeout:
            # 检查进程是否还在运行
            if self.process.poll() is not None:
                output = self.process.stdout.read()
                raise RuntimeError(
                    f"SGLang 服务器启动失败 (退出码: {self.process.returncode})\n"
                    f"最后 50 行输出:\n{output[-2000:]}"
                )
            
            # 打印进度
            current_time = time.time()
            if current_time - last_log_time > 5:
                elapsed = int(current_time - start_time)
                print(f"   等待中... ({elapsed}s)")
                last_log_time = current_time
            
            # 检查健康状态
            try:
                response = requests.get(f"{self.url}/health", timeout=2)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        # 超时，读取输出
        output_lines = []
        try:
            for line in self.process.stdout:
                output_lines.append(line)
                if len(output_lines) > 50:
                    output_lines.pop(0)
        except:
            pass
        
        raise RuntimeError(
            f"SGLang 服务器启动超时 ({timeout}s)\n"
            f"最后输出:\n{''.join(output_lines[-20:])}"
        )
    
    def generate_tokens(
        self, 
        input_ids: List[int], 
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> List[int]:
        """生成并返回完整的 token 序列"""
        response = requests.post(
            f"{self.url}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            },
            timeout=120
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"SGLang 生成失败: {response.text}")
        
        result = response.json()
        
        if "output_ids" in result:
            return result["output_ids"]
        elif "token_ids" in result:
            return result["token_ids"]
        else:
            raise NotImplementedError(
                "SGLang 未返回 token IDs，需要调整 API 或实现 fallback"
            )
    
    def generate_text(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """基于 messages 生成文本 (OpenAI 格式)"""
        response = requests.post(
            f"{self.url}/v1/chat/completions",
            json={
                "model": self.model_path,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
            },
            timeout=120
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"SGLang 生成失败: {response.text}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def shutdown(self):
        """关闭服务器"""
        if self.process:
            print("🛑 关闭 SGLang 服务器...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            print("✅ SGLang 服务器已关闭")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
