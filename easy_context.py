import argparse
import os
from typing import Any, List

from vllm import LLM, SamplingParams

from uq_shared import load_pickle, save_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create easy-context paraphrases from question + ground truths."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="vLLM model id/path used for generating easy contexts.",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=4)

    parser.add_argument("--data_path", type=str, default="./datasets")
    parser.add_argument("--dataset_name", type=str, default="qasper_train.pkl")
    parser.add_argument("--sample_num", type=int, default=1000)
    parser.add_argument("--start_index", type=int, default=0)

    parser.add_argument("--save_path", type=str, default="./easy_contexts")
    parser.add_argument("--save_name", type=str, default="llama8b_qasper_train_easy_contexts.pkl")
    parser.add_argument(
        "--save_every",
        type=int,
        default=20,
        help="Checkpoint output file every N samples.",
    )

    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_prompt(question: str, ground_truths: Any) -> str:
    return f"""
### INSTRUCTIONS ###
You will be given a question and a list of ground truths. Write a single natural answer in one to two sentences.

### GUIDELINES ###
1. Use only the most relevant ground truth.
2. Do not add labels or explanations.
3. Output only the answer sentence(s).

### INPUT ###
Question: {question}
Ground Truths: {ground_truths}

### YOUR ANSWER ###
"""


def paraphrase_text(question: str, ground_truths: Any, llm: LLM, sampling_params: SamplingParams) -> str:
    prompt = build_prompt(question, ground_truths)
    messages = [{"role": "user", "content": prompt}]
    outputs = llm.chat(messages, sampling_params=sampling_params)
    return outputs[0].outputs[0].text.strip()


def main() -> None:
    args = parse_args()

    data_file = os.path.join(args.data_path, args.dataset_name)
    dataset = load_pickle(data_file)
    if not isinstance(dataset, list):
        raise ValueError(f"Expected list dataset in {data_file}, got {type(dataset)}")

    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
    )
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)

    end_index = min(len(dataset), args.start_index + args.sample_num)
    subset = dataset[args.start_index:end_index]

    os.makedirs(args.save_path, exist_ok=True)
    save_file = os.path.join(args.save_path, args.save_name)

    generated_contexts: List[str] = []
    for idx, row in enumerate(subset, start=1):
        question = row["question"]
        ground_truths = row["ground_truths"]
        text = paraphrase_text(question, ground_truths, llm, sampling_params)
        generated_contexts.append(text)

        if args.save_every > 0 and idx % args.save_every == 0:
            save_pickle(save_file, generated_contexts)
            print(f"Checkpoint saved at {idx}/{len(subset)} -> {save_file}")

    save_pickle(save_file, generated_contexts)
    print(f"Saved {len(generated_contexts)} easy contexts to {save_file}")


if __name__ == "__main__":
    main()
