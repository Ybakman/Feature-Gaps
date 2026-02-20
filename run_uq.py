import argparse
import os

import TruthTorchLM as ttlm
from TruthTorchLM.evaluators.eval_truth_method import get_metric_scores

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DebertaForSequenceClassification,
    DebertaTokenizer,
)

from uq_shared import load_pickle, resolve_model_id, save_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TruthTorchLM uncertainty methods.")
    parser.add_argument("--model_path", type=str, default="../../model-registry")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--auto_device", action="store_true", help="Use automatic device mapping.")
    parser.add_argument("--quantize", action="store_true", help="Load model in 8-bit mode.")

    parser.add_argument("--data_path", type=str, default="../results")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="../uq_results")
    parser.add_argument("--save_name", type=str, required=True)

    parser.add_argument("--number_of_generations", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--entailment_model", type=str, default="microsoft/deberta-large-mnli")
    parser.add_argument("--entailment_device", type=str, default="cuda:0")
    parser.add_argument("--eval_metrics", nargs="+", default=["auroc", "prr"])

    return parser.parse_args()


def load_generation_model(args: argparse.Namespace):
    model_id = resolve_model_id(args.model_name, args.model_path)

    if args.auto_device:
        if args.quantize:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="balanced_low_0",
                quantization_config=quantization_config,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
    else:
        target_device = torch.device(f"cuda:{args.device}")
        if args.quantize:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map={"": target_device},
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None and tokenizer.default_chat_template is not None:
        tokenizer.chat_template = tokenizer.default_chat_template

    return model, tokenizer


def build_truth_methods(args: argparse.Namespace, model, tokenizer, ttlm):
    entailment_model = DebertaForSequenceClassification.from_pretrained(args.entailment_model).to(
        args.entailment_device
    )
    entailment_tokenizer = DebertaTokenizer.from_pretrained(args.entailment_model)

    length_norm = ttlm.scoring_methods.LengthNormalizedScoring()

    return [
        ttlm.truth_methods.SemanticEntropy(
            length_norm,
            number_of_generations=args.number_of_generations,
            model_for_entailment=entailment_model,
            tokenizer_for_entailment=entailment_tokenizer,
        ),
        ttlm.truth_methods.Confidence(length_norm),
        ttlm.truth_methods.Entropy(length_norm, number_of_generations=args.number_of_generations),
        ttlm.truth_methods.EccentricityUncertainty(
            number_of_generations=args.number_of_generations,
            model_for_entailment=entailment_model,
            tokenizer_for_entailment=entailment_tokenizer,
        ),
        ttlm.truth_methods.KernelLanguageEntropy(
            number_of_generations=args.number_of_generations,
            model_for_entailment=entailment_model,
            tokenizer_for_entailment=entailment_tokenizer,
        ),
        ttlm.truth_methods.EccentricityConfidence(
            number_of_generations=args.number_of_generations,
            model_for_entailment=entailment_model,
            tokenizer_for_entailment=entailment_tokenizer,
        ),
        ttlm.truth_methods.ContextCheck(
            check_model=model,
            check_tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        ),
        ttlm.truth_methods.MARS(device=torch.device(f"cuda:{args.device}"), mars_temperature=0.1),
        ttlm.truth_methods.MiniCheckMethod(minicheck_model="flan-t5-large"),
        ttlm.truth_methods.SAR(
            number_of_generations=args.number_of_generations,
            similarity_model_device=torch.device(f"cuda:{args.device}"),
        ),
        ttlm.truth_methods.MatrixDegreeUncertainty(
            number_of_generations=args.number_of_generations,
            model_for_entailment=entailment_model,
            tokenizer_for_entailment=entailment_tokenizer,
        ),
        ttlm.truth_methods.MatrixDegreeConfidence(
            number_of_generations=args.number_of_generations,
            model_for_entailment=entailment_model,
            tokenizer_for_entailment=entailment_tokenizer,
        ),
        ttlm.truth_methods.SumEigenUncertainty(
            number_of_generations=args.number_of_generations,
            model_for_entailment=entailment_model,
            tokenizer_for_entailment=entailment_tokenizer,
        ),
    ]


def main() -> None:
    args = parse_args()



    model, tokenizer = load_generation_model(args)
    truth_methods = build_truth_methods(args, model, tokenizer, ttlm)

    dataset_path = os.path.join(args.data_path, args.dataset_name)
    output_dict = load_pickle(dataset_path)

    output_dict = ttlm.utils.eval_utils.run_truth_methods_over_dataset(
        output_dict,
        model,
        truth_methods=truth_methods,
        tokenizer=tokenizer,
        return_method_details=True,
        seed=args.seed,
        batch_generation=False,
    )

    eval_list = get_metric_scores(
        output_dict=output_dict,
        eval_metrics=args.eval_metrics,
        seed=args.seed,
    )

    results = {"eval_list": eval_list, "output_dict": output_dict}

    for idx, metrics in enumerate(eval_list):
        print(f"{output_dict['truth_methods'][idx]} -> {metrics}")

    save_file = os.path.join(args.save_path, f"{args.save_name}.pkl")
    save_pickle(save_file, results)
    print(f"Saved run_uq results to {save_file}")


if __name__ == "__main__":
    main()
