import argparse
import logging
import threading
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: str
    content: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 0.7
    max_tokens: int | None = 512
    stream: bool | None = False
    top_p: float | None = 1.0


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


MODEL_LOCK = threading.Lock()
model_cache: dict[str, Any] = {}


def get_model(model_name: str, device: str = "cuda"):
    with MODEL_LOCK:
        if model_name not in model_cache:
            logger.info(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                trust_remote_code=True,
                torch_dtype="auto",
            )
            model.eval()
            model_cache[model_name] = {"model": model, "tokenizer": tokenizer}
        return model_cache[model_name]


def generate_response(
    model: Any,
    tokenizer: Any,
    messages: list[ChatMessage],
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 1.0,
) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    return response


app = FastAPI(title="Mini-LLM-Server")


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "default",
                "object": "model",
                "created": 1234567890,
                "owned_by": "mini-agentic-rl",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    import time

    model_info = get_model(request.model)
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]

    try:
        response_text = generate_response(
            model,
            tokenizer,
            request.messages,
            request.temperature or 0.7,
            request.max_tokens or 512,
            request.top_p or 1.0,
        )

        prompt_tokens = sum(len(tokenizer.encode(m.content)) for m in request.messages)
        completion_tokens = len(tokenizer.encode(response_text))

        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}


def main():
    parser = argparse.ArgumentParser(description="Mini-LLM-Server")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2-0.5B", help="Model name or path"
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    logger.info(f"Starting Mini-LLM-Server with model: {args.model}")
    get_model(args.model, args.device)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
