import argparse
import os
from typing import List

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

from uq_shared import load_pickle, resolve_model_id, save_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LookbackLens over generated outputs.")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-instruct-v0.3")
    parser.add_argument("--model_path", type=str, default="../../model-registry")
    parser.add_argument("--torch_dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device_map", type=str, default="auto")

    parser.add_argument("--results_path", type=str, default="../results")
    parser.add_argument("--train_results_file", type=str, required=True)
    parser.add_argument("--test_results_file", type=str, required=True)
    parser.add_argument("--train_num_samples", type=int, default=128)
    parser.add_argument("--selection_mode", choices=["shortest_context", "head"], default="shortest_context")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="../uq_results")
    parser.add_argument("--save_name", type=str, default="lookbacklens_results.pkl")
    return parser.parse_args()


def parse_dtype(dtype_name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def select_indices(results: dict, num_samples: int, mode: str) -> List[int]:
    if num_samples == -1 or num_samples >= len(results["contexts"]):
        return list(range(len(results["contexts"])))
    if mode == "head":
        return list(range(num_samples))

    lengths = [len(ctx) for ctx in results["contexts"]]
    ranked = sorted(range(len(lengths)), key=lambda idx: lengths[idx])
    return ranked[:num_samples]


def extract_attention_features(model, tokenizer, generation_dict: dict) -> np.ndarray:
    with torch.no_grad():
        all_ids = generation_dict["model_output"].to(model.device)
        text = generation_dict["text"]
        prompt_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)

        output = model(all_ids, output_attentions=True)
        n_layers = len(output.attentions)
        n_heads = output.attentions[0].shape[1]
        features = np.zeros((n_layers, n_heads), dtype=np.float32)

        context_len = min(prompt_ids.shape[1], output.attentions[0].shape[-1])
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                attention_matrix = output.attentions[layer_idx][0, head_idx]
                last_row = attention_matrix[-1]

                context_mean = torch.mean(last_row[:context_len]).item()
                answer_slice = last_row[context_len:]
                answer_mean = torch.mean(answer_slice).item() if answer_slice.numel() > 0 else 0.0
                features[layer_idx, head_idx] = context_mean / (context_mean + answer_mean + 1e-12)
    return features.reshape(-1)


def build_feature_matrix(model, tokenizer, results: dict, indices: List[int]) -> np.ndarray:
    features = []
    for idx in indices:
        features.append(extract_attention_features(model, tokenizer, results["generation_dicts"][idx]))
    return np.nan_to_num(np.array(features), nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    args = parse_args()
    import TruthTorchLM as ttlm

    train_results = load_pickle(os.path.join(args.results_path, args.train_results_file))
    test_results = load_pickle(os.path.join(args.results_path, args.test_results_file))

    model_id = resolve_model_id(args.model_name, args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=parse_dtype(args.torch_dtype),
        device_map=args.device_map,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_indices = select_indices(train_results, args.train_num_samples, args.selection_mode)
    train_features = build_feature_matrix(model, tokenizer, train_results, train_indices)
    train_labels = np.array([train_results["generations_correctness"][idx] for idx in train_indices])

    valid_mask = train_labels != -1
    clf = LogisticRegression(random_state=args.seed, max_iter=200)
    clf.fit(train_features[valid_mask], train_labels[valid_mask])

    train_scores = clf.predict_proba(train_features)[:, 1]
    train_metrics = ttlm.utils.metric_score(["prr", "auroc"], train_labels, train_scores, train_scores)
    print(f"Train metrics: {train_metrics}")

    test_indices = list(range(len(test_results["generation_dicts"])))
    test_features = build_feature_matrix(model, tokenizer, test_results, test_indices)
    test_labels = np.array(test_results["generations_correctness"])

    test_scores = clf.predict_proba(test_features)[:, 1]
    test_metrics = ttlm.utils.metric_score(["prr", "auroc"], test_labels, test_scores, test_scores)
    print(f"Test metrics: {test_metrics}")

    results = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_indices": train_indices,
        "test_scores": test_scores,
        "train_scores": train_scores,
        "train_results_file": args.train_results_file,
        "test_results_file": args.test_results_file,
    }

    save_file = os.path.join(args.output_path, args.save_name)
    save_pickle(save_file, results)
    print(f"Saved LookbackLens results to {save_file}")


if __name__ == "__main__":
    main()
