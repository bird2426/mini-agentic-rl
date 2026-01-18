"""
交互式测试脚本
测试训练好的模型
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="交互式测试")
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型路径")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="温度")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("交互式测试")
    print("=" * 80)
    print(f"模型: {args.model_path}")
    print(f"加载中...")
    
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    
    print("✅ 模型加载完成")
    print("\n输入问题进行测试（输入 'quit' 退出）：")
    print("=" * 80)
    
    # 交互循环
    while True:
        try:
            # 获取用户输入
            user_input = input("\n💬 你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            
            # 构造消息
            messages = [
                {"role": "user", "content": user_input}
            ]
            
            # 应用 chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # 生成
            print("🤖 助手: ", end="", flush=True)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # 解码（只取新生成的部分）
            input_length = inputs['input_ids'].shape[1]
            response = tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )
            
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            continue


if __name__ == "__main__":
    main()
