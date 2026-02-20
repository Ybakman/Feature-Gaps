import argparse
import os
from typing import Dict, List

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM

from uq_shared import load_pickle, resolve_model_id, save_pickle, subset_hidden_state_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature_gaps method over hidden state dumps.")
    parser.add_argument("--hidden_states_path", type=str, default="../hidden_states")

    parser.add_argument("--train_files", nargs="+", required=True)
    parser.add_argument("--val_files", nargs="+", required=True)
    parser.add_argument("--test_files", nargs="+", required=True)
    parser.add_argument(
        "--feature_names",
        nargs="+",
        default=None,
        help="Optional names for each file triplet. Must match --train_files length.",
    )

    parser.add_argument("--train_num", type=int, default=256)
    parser.add_argument("--train_start_index", type=int, default=0)
    parser.add_argument("--val_num", type=int, default=256)
    parser.add_argument("--val_start_index", type=int, default=0)
    parser.add_argument("--test_num", type=int, default=-1)
    parser.add_argument("--test_start_index", type=int, default=0)

    parser.add_argument("--pca_mode", choices=["avg", "first"], default="avg")
    parser.add_argument("--val_mode", choices=["avg", "first"], default="avg")
    parser.add_argument("--test_mode", choices=["all", "first"], default="all")
    parser.add_argument("--hidden_state_mode", choices=["naive", "project", "prob"], default="naive")
    parser.add_argument("--ignore_first_layers", type=int, default=7)

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model_path", type=str, default="../../model-registry")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="../uq_results")
    parser.add_argument("--save_name", type=str, default="feature_gaps_results.pkl")
    return parser.parse_args()


def _sorted_layers(layer_map: Dict) -> List:
    return sorted(layer_map.keys(), key=int)


def _score_direction(state: np.ndarray, mean: np.ndarray, direction: np.ndarray) -> float:
    denom = np.linalg.norm(direction) + 1e-12
    return float(np.dot(state - mean, direction) / denom)


def _build_directions(train_result: Dict, pca_mode: str) -> Dict:
    if pca_mode == "avg":
        perturbed = train_result["hidden_states_negative_avg"]
        regular = train_result["hidden_states_positive_avg"]
    else:
        perturbed = train_result["hidden_states_negative_first"]
        regular = train_result["hidden_states_positive_first"]

    diff_map: Dict = {}
    for perturb_mode in train_result["perturb_modes"]:
        layer_map = perturbed[perturb_mode]
        for layer in layer_map.keys():
            diffs = []
            for idx in range(len(layer_map[layer])):
                diffs.append((regular[layer][idx] - layer_map[layer][idx]).reshape(1, -1))
            if layer not in diff_map:
                diff_map[layer] = diffs
            else:
                diff_map[layer].extend(diffs)

    directions = {}
    for layer in diff_map.keys():
        diffs = np.concatenate(diff_map[layer], axis=0)
        mean_vector = np.mean(diffs, axis=0)
        centered = diffs - mean_vector.reshape(1, -1)
        pca = PCA(n_components=1, whiten=False, random_state=0).fit(centered)
        directions[layer] = {"mean": mean_vector, "direction": pca.components_[0]}
    return directions


def _get_positive_hidden_states(result: Dict, mode: str):
    if mode == "avg":
        return result["hidden_states_positive_avg"]
    return result["hidden_states_positive_first"]


def _compute_validation_scores(result: Dict, directions: Dict, val_mode: str) -> Dict:
    hidden_states = _get_positive_hidden_states(result, val_mode)
    scores_by_layer = {}
    for layer in _sorted_layers(hidden_states):
        state_list = hidden_states[layer]
        mean = directions[layer]["mean"]
        direction = directions[layer]["direction"]
        scores_by_layer[layer] = [_score_direction(state, mean, direction) for state in state_list]
    return scores_by_layer


def _get_test_hidden_states(result: Dict, test_mode: str, hidden_state_mode: str):
    if test_mode == "all":
        if hidden_state_mode == "naive":
            return result["hidden_states_positive_avg"]
        return result["hidden_states_positive_all"]
    return result["hidden_states_positive_first"]


def _compute_test_scores(
    states: List,
    mean: np.ndarray,
    direction: np.ndarray,
    hidden_state_mode: str,
    lm_head_projection: torch.Tensor | None,
    topk_indices: List | None,
    topk_values: List | None,
) -> np.ndarray:
    raw_scores = []
    for idx, state in enumerate(states):
        base_score = _score_direction(state, mean, direction)
        if hidden_state_mode == "naive":
            raw_scores.append(base_score)
            continue

        if lm_head_projection is None:
            raise ValueError("lm_head projection is required for project/prob modes.")

        if hidden_state_mode == "project":
            raw_scores.append(float(torch.mean(lm_head_projection * base_score).item()))
            continue

        token_indices = topk_indices[idx].to(dtype=torch.long)
        token_values = topk_values[idx]
        selected_projection = lm_head_projection[token_indices]
        weighted = selected_projection * token_values
        raw_scores.append(float((weighted.mean() * base_score).item()))
    return np.array(raw_scores)


def main() -> None:
    args = parse_args()
    import TruthTorchLM as ttlm

    if not (len(args.train_files) == len(args.val_files) == len(args.test_files)):
        raise ValueError("--train_files, --val_files and --test_files must have the same length.")

    feature_count = len(args.train_files)
    if args.feature_names is None:
        feature_names = [f"feature_{idx}" for idx in range(feature_count)]
    else:
        if len(args.feature_names) != feature_count:
            raise ValueError("--feature_names length must match --train_files length.")
        feature_names = args.feature_names

    train_results = []
    val_results = []
    test_results = []
    for train_file, val_file, test_file in zip(args.train_files, args.val_files, args.test_files):
        train_result = load_pickle(os.path.join(args.hidden_states_path, train_file))
        val_result = load_pickle(os.path.join(args.hidden_states_path, val_file))
        test_result = load_pickle(os.path.join(args.hidden_states_path, test_file))

        train_results.append(subset_hidden_state_result(train_result, args.train_num, args.train_start_index))
        val_results.append(subset_hidden_state_result(val_result, args.val_num, args.val_start_index))
        test_results.append(subset_hidden_state_result(test_result, args.test_num, args.test_start_index))

    lm_head_projection = None
    if args.hidden_state_mode in {"project", "prob"}:
        model_id = resolve_model_id(args.model_name, args.model_path)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        first_layer = _sorted_layers(_get_positive_hidden_states(test_results[0], "avg"))[0]
        hidden_dim = _get_positive_hidden_states(test_results[0], "avg")[first_layer][0].shape[-1]
        direction_probe = torch.zeros((1, hidden_dim), dtype=torch.float16, device=model.device)
        lm_head_projection = model.lm_head(direction_probe)[0].detach().cpu()

    all_directions = []
    all_signs = []
    all_layer_scores = []
    best_layers = []
    best_prrs = []
    best_aurocs = []

    for feature_idx in range(feature_count):
        directions = _build_directions(train_results[feature_idx], args.pca_mode)
        all_directions.append(directions)

        val_scores_by_layer = _compute_validation_scores(val_results[feature_idx], directions, args.val_mode)
        correctness = np.array(val_results[feature_idx]["correctness"])

        signs_by_layer = {}
        prr_by_layer = {}
        auroc_by_layer = {}

        for layer in _sorted_layers(val_scores_by_layer):
            raw_scores = np.array(val_scores_by_layer[layer])
            if int(layer) < args.ignore_first_layers:
                signs_by_layer[layer] = 1
                prr_by_layer[layer] = 0.0
                auroc_by_layer[layer] = 0.5
                continue

            raw_auroc = ttlm.utils.metric_score(["auroc"], correctness, raw_scores, raw_scores)["auroc"]
            sign = -1 if raw_auroc < 0.5 else 1
            signed_scores = raw_scores * sign
            evals = ttlm.utils.metric_score(["auroc", "prr"], correctness, signed_scores, signed_scores)
            signs_by_layer[layer] = sign
            prr_by_layer[layer] = evals["prr"]
            auroc_by_layer[layer] = evals["auroc"]

        best_layer = max(prr_by_layer, key=prr_by_layer.get)
        best_layers.append(best_layer)
        best_prrs.append(prr_by_layer[best_layer])
        best_aurocs.append(auroc_by_layer[best_layer])
        all_signs.append(signs_by_layer)
        all_layer_scores.append(val_scores_by_layer)
        print(
            f"Validation {feature_names[feature_idx]}: layer={best_layer} "
            f"auroc={auroc_by_layer[best_layer]:.4f} prr={prr_by_layer[best_layer]:.4f}"
        )

    val_feature_scores = []
    for feature_idx in range(feature_count):
        val_feature_scores.append(np.array(all_layer_scores[feature_idx][best_layers[feature_idx]]))
    val_feature_scores = np.array(val_feature_scores).T

    val_labels = np.array(val_results[0]["correctness"])
    valid_mask = val_labels != -1
    clf_best_scores = LogisticRegression(random_state=args.seed, max_iter=200)
    clf_best_scores.fit(val_feature_scores[valid_mask], val_labels[valid_mask])
    val_pred = clf_best_scores.predict_proba(val_feature_scores)[:, 1]
    ensemble_val_metrics = ttlm.utils.metric_score(["prr", "auroc"], val_labels, val_pred, val_pred)
    print(f"Validation ensemble metrics: {ensemble_val_metrics}")

    test_feature_scores = []
    individual_test_metrics = []

    for feature_idx in range(feature_count):
        best_layer = best_layers[feature_idx]
        best_direction = all_directions[feature_idx][best_layer]
        test_hidden_states = _get_test_hidden_states(
            test_results[feature_idx], args.test_mode, args.hidden_state_mode
        )

        topk_indices = test_results[feature_idx].get("topk_indices")
        topk_values = test_results[feature_idx].get("topk_values")
        raw_scores = _compute_test_scores(
            test_hidden_states[best_layer],
            best_direction["mean"],
            best_direction["direction"],
            args.hidden_state_mode,
            lm_head_projection,
            topk_indices,
            topk_values,
        )
        test_feature_scores.append(raw_scores)

        signed_scores = raw_scores * all_signs[feature_idx][best_layer]
        correctness_test = np.array(test_results[feature_idx]["correctness"])
        evals = ttlm.utils.metric_score(["auroc", "prr"], correctness_test, signed_scores, signed_scores)
        individual_test_metrics.append(evals)
        print(f"Test {feature_names[feature_idx]}: {evals}")

    test_feature_scores = np.array(test_feature_scores).T
    test_labels = np.array(test_results[0]["correctness"])
    test_pred = clf_best_scores.predict_proba(test_feature_scores)[:, 1]
    ensemble_test_metrics = ttlm.utils.metric_score(["prr", "auroc"], test_labels, test_pred, test_pred)
    print(f"Test ensemble metrics: {ensemble_test_metrics}")

    results = {
        "feature_names": feature_names,
        "best_layers": best_layers,
        "best_prrs": best_prrs,
        "best_aurocs": best_aurocs,
        "individual_test_metrics": individual_test_metrics,
        "ensemble_validation_metrics": ensemble_val_metrics,
        "ensemble_test_metrics": ensemble_test_metrics,
        "test_scores": test_feature_scores,
        "val_scores": val_feature_scores,
    }

    save_file = os.path.join(args.output_path, args.save_name)
    save_pickle(save_file, results)
    print(f"Saved feature_gaps results to {save_file}")


if __name__ == "__main__":
    main()
