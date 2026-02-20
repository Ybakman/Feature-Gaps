import os
import pickle
from pathlib import Path
from typing import Any, Dict


def resolve_model_id(model_name: str, model_path: str) -> str:
    candidate = Path(model_path) / model_name
    if candidate.exists():
        return str(candidate)
    return model_name


def load_pickle(path: str) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_pickle(path: str, obj: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(obj, handle)


def _slice_list(values, num: int, start_index: int):
    if num == -1:
        return values[start_index:]
    end_index = start_index + num
    return values[start_index:end_index]


def subset_hidden_state_result(
    result: Dict[str, Any], num: int, start_index: int = 0
) -> Dict[str, Any]:
    if num == -1 and start_index == 0:
        return result

    key_groups = [
        "questions",
        "correctness",
        "generation_dicts",
        "contexts",
        "perturbed_contexts",
        "topk_indices",
        "topk_values",
    ]
    for key in key_groups:
        if key in result and isinstance(result[key], list):
            result[key] = _slice_list(result[key], num, start_index)

    for key in ["hidden_states_positive_avg", "hidden_states_positive_first", "hidden_states_positive_all"]:
        if key in result and isinstance(result[key], dict):
            for layer in list(result[key].keys()):
                result[key][layer] = _slice_list(result[key][layer], num, start_index)

    for key in ["hidden_states_negative_avg", "hidden_states_negative_first", "hidden_states_negative_all"]:
        if key in result and isinstance(result[key], dict):
            for perturb_mode in list(result[key].keys()):
                layer_map = result[key][perturb_mode]
                for layer in list(layer_map.keys()):
                    layer_map[layer] = _slice_list(layer_map[layer], num, start_index)

    return result


def build_hidden_state_file(hidden_states_path: str, name: str) -> str:
    return os.path.join(hidden_states_path, name)
