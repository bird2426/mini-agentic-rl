import uvicorn
import torch
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List

class HFModelServer:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        dtype = torch.float16 if device == "mps" or device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device)
        self.model.eval()
        
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            try:
                data = await request.json()
                messages = data.get("messages", [])
                max_tokens = data.get("max_tokens", 256)
                temperature = data.get("temperature", 0.7)

                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                completion_ids = outputs[0][input_ids.shape[1]:]
                completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": completion_text
                            }
                        }
                    ]
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    server = HFModelServer(args.model_path, args.device)
    server.run(port=args.port)
