import argparse
import os
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression

from uq_shared import load_pickle, save_pickle, subset_hidden_state_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAPLMA over extracted hidden states.")
    parser.add_argument("--hidden_states_path", type=str, default="../hidden_states")

    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)

    parser.add_argument("--train_num", type=int, default=128)
    parser.add_argument("--train_start_index", type=int, default=0)
    parser.add_argument("--val_num", type=int, default=128)
    parser.add_argument("--val_start_index", type=int, default=128)
    parser.add_argument("--test_num", type=int, default=-1)
    parser.add_argument("--test_start_index", type=int, default=0)

    parser.add_argument("--token_mode", type=str, choices=["first", "last", "avg"], default="last")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="../uq_results")
    parser.add_argument("--save_name", type=str, default="saplma_results.pkl")
    return parser.parse_args()


def to_hidden_state_features(layer_states: List[np.ndarray], token_mode: str) -> np.ndarray:
    features = []
    for sample in layer_states:
        if token_mode == "last":
            features.append(sample[-1])
        elif token_mode == "first":
            features.append(sample[0])
        else:
            features.append(np.mean(sample, axis=0))
    return np.array(features)


def evaluate_layer(ttlm, correctness: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return ttlm.utils.metric_score(["prr", "auroc"], correctness, scores, scores)


def main() -> None:
    args = parse_args()
    import TruthTorchLM as ttlm

    train_result = load_pickle(os.path.join(args.hidden_states_path, args.train_file))
    val_result = load_pickle(os.path.join(args.hidden_states_path, args.val_file))
    test_result = load_pickle(os.path.join(args.hidden_states_path, args.test_file))

    train_result = subset_hidden_state_result(train_result, args.train_num, args.train_start_index)
    val_result = subset_hidden_state_result(val_result, args.val_num, args.val_start_index)
    test_result = subset_hidden_state_result(test_result, args.test_num, args.test_start_index)

    hidden_states_train = train_result["hidden_states_positive_all"]
    correctness_train = np.array(train_result["correctness"])
    labels_train = correctness_train[correctness_train != -1]

    all_classifiers = {}
    for layer in hidden_states_train.keys():
        layer_features = to_hidden_state_features(hidden_states_train[layer], args.token_mode)
        clean_features = layer_features[correctness_train != -1]
        clf = LogisticRegression(random_state=args.seed, max_iter=200)
        clf.fit(clean_features, labels_train)
        all_classifiers[layer] = clf

    hidden_states_val = val_result["hidden_states_positive_all"]
    correctness_val = np.array(val_result["correctness"])

    best_layer = None
    best_prr = float("-inf")
    best_evals = None

    for layer, clf in all_classifiers.items():
        layer_features = to_hidden_state_features(hidden_states_val[layer], args.token_mode)
        predicted_scores = clf.predict_proba(layer_features)[:, 1]
        evals = evaluate_layer(ttlm, correctness_val, predicted_scores)
        if evals["prr"] > best_prr:
            best_prr = evals["prr"]
            best_layer = layer
            best_evals = evals

    hidden_states_test = test_result["hidden_states_positive_all"][best_layer]
    test_features = to_hidden_state_features(hidden_states_test, args.token_mode)
    correctness_test = np.array(test_result["correctness"])
    predicted_scores_test = all_classifiers[best_layer].predict_proba(test_features)[:, 1]
    test_evals = evaluate_layer(ttlm, correctness_test, predicted_scores_test)

    print(f"Best validation layer: {best_layer} -> {best_evals}")
    print(f"Test metrics: {test_evals}")

    results = {
        "best_layer": best_layer,
        "validation_metrics": best_evals,
        "test_metrics": test_evals,
        "predicted_scores_test": predicted_scores_test,
        "token_mode": args.token_mode,
        "train_file": args.train_file,
        "val_file": args.val_file,
        "test_file": args.test_file,
    }

    save_file = os.path.join(args.output_path, args.save_name)
    save_pickle(save_file, results)
    print(f"Saved SAPLMA results to {save_file}")


if __name__ == "__main__":
    main()
