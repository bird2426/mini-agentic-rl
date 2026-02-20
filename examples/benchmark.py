"""
Benchmark - 评估模型在 GSM8K 上的准确率

只评估模型,不包含训练
"""

import argparse
import logging
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


COT_PROMPT = """Solve the following math problem step by step.

Example 1:
Problem: There are 5 birds on a tree. 2 fly away. How many birds are left?
Solution: 5 - 2 = 3. There are 3 birds left. #### 3

Example 2:
Problem: Tom has 10 apples. He gives 3 to Jerry. How many apples does Tom have now?
Solution: 10 - 3 = 7. Tom has 7 apples now. #### 7

Example 3:
Problem: Lisa has 8 candies. She buys 5 more. How many candies does she have in total?
Solution: 8 + 5 = 13. Lisa has 13 candies in total. #### 13

Now solve this problem:
Problem: {problem}

Solution:"""


def extract_answer(response: str):
    response = response.strip()

    if "####" in response:
        for line in reversed(response.split("\n")):
            if "####" in line:
                return line.split("####")[-1].strip()

    response = re.sub(r"```[\s\S]*?```", "", response)

    match = re.search(
        r"(?:final answer|answer is|result is|=|:)\s*(\d+)", response, re.IGNORECASE
    )
    if match:
        return match.group(1)

    match = re.search(r"\d+\s*[-+\*/]\s*\d+\s*=\s*(\d+)", response)
    if match:
        return match.group(1)

    numbers = re.findall(r"\d+", response)
    if numbers:
        return numbers[-1]

    return None


def load_gsm8k(split="test", num_samples=100):
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if num_samples:
        dataset = dataset.select(range(num_samples))
    return dataset


def evaluate(model, tokenizer, device, dataset):
    correct = 0
    total = len(dataset)

    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["answer"]

        answer_str = (
            ground_truth.split("####")[-1].strip()
            if "####" in ground_truth
            else ground_truth
        )
        try:
            ground_num = int(answer_str.replace(",", "").strip())
        except:
            ground_num = answer_str

        prompt = COT_PROMPT.format(problem=question)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
            )
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        pred = extract_answer(response)

        is_correct = False
        if pred:
            try:
                pred_clean = pred.replace(",", "").strip()
                if pred_clean == str(ground_num):
                    is_correct = True
                elif abs(int(pred_clean) - int(ground_num)) <= 1:
                    is_correct = True
            except:
                pass

        if is_correct:
            correct += 1

        if (i + 1) % 20 == 0:
            logger.info(
                f"Progress: {i + 1}/{total}, Accuracy: {correct / (i + 1) * 100:.1f}%"
            )

    accuracy = correct / total * 100
    return accuracy, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument(
        "--adapter", type=str, default=None, help="Path to LoRA adapter"
    )
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Loading model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    if args.adapter:
        logger.info(f"Loading adapter from: {args.adapter}")
        from peft import PeftModel

        model = model.to("cpu")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.to(device)

    logger.info(f"Loading GSM8K {args.split} split ({args.num_samples} samples)...")
    dataset = load_gsm8k(split=args.split, num_samples=args.num_samples)

    logger.info("Evaluating...")
    accuracy, correct, total = evaluate(model, tokenizer, device, dataset)

    logger.info("=" * 60)
    logger.info("GSM8K Benchmark Results")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Samples: {total}")
    logger.info(f"Correct: {correct}")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
