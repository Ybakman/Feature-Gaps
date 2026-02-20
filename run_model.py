import argparse
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from uq_shared import load_pickle, save_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark outputs over a dataset.")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_path", type=str, default="./datasets")
    parser.add_argument("--dataset_name", type=str, default="qasper.pkl")
    parser.add_argument("--dataset_size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--save_name", type=str, default="qasper_15")

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--auto_device", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--api_env_file", type=str, default="./api_values.env")

    parser.add_argument("--model_judge", type=str, default="gemini-2.5-flash-lite")
    parser.add_argument("--judge_retries", type=int, default=30)

    return parser.parse_args()


def load_generation_model(args: argparse.Namespace):
    if args.auto_device:
        if args.quantize:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                device_map="balanced_low_0",
                quantization_config=quantization_config,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16,
                device_map="balanced_low_0",
            )
    else:
        target_device = torch.device(f"cuda:{args.device}")
        if args.quantize:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=quantization_config,
                device_map={"": target_device},
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16,
                device_map="balanced_low_0",
            )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None and tokenizer.default_chat_template is not None:
        tokenizer.chat_template = tokenizer.default_chat_template

    return model, tokenizer


def build_evaluator(args: argparse.Namespace, ttlm):
    judge_prompt = """
You will be given a context, a question relevant to that context, a predicted answer, and a list of possible ground truth answers provided by human experts. Your task is to assign one of the following labels: ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
Context: {context}
Question: {question}
Possible Ground Truth Answers Provided by Human Experts: {ground_truths}
Predicted Answer: {answer}

Labeling instructions:
- Assign "NOT_ATTEMPTED" if the predicted answer fails to engage with the question, or explicitly states that an answer is not found in the context.
- Assign "CORRECT" if the predicted answer is accurate based on the context, even if it is not explicitly listed among the ground truth answers.
- Assign "INCORRECT" if the predicted answer is contradicted by the context or wrong.

Important: The ground truth list may not cover all valid answers. For those cases, look at the context.

Provide your explanation and then at the end give your grade.

Response:"""
    return ttlm.evaluators.ModelJudge(
        args.model_judge,
        num_retries=args.judge_retries,
        prompt=judge_prompt,
    )


def load_api_env_file(path: str) -> None:
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()


def main() -> None:
    args = parse_args()
    import TruthTorchLM as ttlm

    load_api_env_file(args.api_env_file)
    model, tokenizer = load_generation_model(args)

    data_file = os.path.join(args.data_path, args.dataset_name)
    dataset = load_pickle(data_file)
    if not isinstance(dataset, list):
        raise ValueError(f"Expected list dataset in {data_file}, got {type(dataset)}")

    random.seed(args.seed)
    sample_size = min(args.dataset_size, len(dataset))
    dataset = random.sample(dataset, sample_size)
    evaluator = build_evaluator(args, ttlm)

    prompt_template = (
        "Context: {context}. Here is a given context. You are a helpful assistant. "
        "Answer the following question with a brief single but complete answer. "
        "Use the context information to answer this question. "
        "Question: {question} Answer:"
    )

    results = ttlm.utils.eval_utils.eval_model_over_dataset(
        dataset,
        model,
        tokenizer=tokenizer,
        correctness_evaluator=evaluator,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=args.do_sample,
        seed=args.seed,
        user_prompt=prompt_template,
    )

    save_file = os.path.join(args.save_path, f"{args.save_name}.pkl")
    save_pickle(save_file, results)
    print(f"Saved run_model output to {save_file}")


if __name__ == "__main__":
    main()
