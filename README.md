# Uncertainty as Feature Gaps: Epistemic Uncertainty Quantification of LLMs in Contextual Question-Answering ICLR-2026

![Feature-Gaps Overview](assets/feature_gaps_overview.png)

Official code for the paper experiments on uncertainty estimation in RAG-style QA using:
- baseline TruthTorchLM UQ methods,
- `feature_gaps`,
- SAPLMA,
- LookbackLens.

This repository is organized as a strict step-by-step pipeline.

## Repository Layout

- `run_model.py`: generates base model outputs + correctness labels.
- `run_uq.py`: runs TruthTorchLM uncertainty methods over step-1 output.
- `easy_context.py`: generates easy-context perturbation text.
- `extract_hidden_states.py`: produces hidden-state artifacts for downstream methods.
- `run_feature_gaps.py`: runs the feature-gaps method.
- `run_saplma.py`: runs SAPLMA.
- `run_lookbacklens.py`: runs LookbackLens.
- `datasets/`: input datasets.
- `results/`, `hidden_states/`, `easy_contexts/`, `uq_results/`: generated outputs.

## Conda Setup (From Scratch)

### Option A: One-command setup (recommended)

```bash
git clone https://github.com/Ybakman/Feature-Gaps.git
cd Feature-Gaps
bash setup_conda.sh
conda activate feature-gaps
mkdir -p results uq_results hidden_states easy_contexts
```

Notes:
- `setup_conda.sh` creates a new conda env and installs all required packages.
- Default env name is `feature-gaps`.
- To change env name: `ENV_NAME=myenv bash setup_conda.sh`
- To install CPU-only PyTorch: `WITH_CUDA=0 bash setup_conda.sh`

Fill `api_values.env` before running `run_model.py`:

```env
OPENAI_API_KEY=...
GEMINI_API_KEY=...
GOOGLE_CLOUD_PROJECT=...
HF_HOME=/home/yavuz/yavuz/.cache/huggingface/
```

## Section A: Run Model (Step 1)

Purpose:
- Generates dataset-level outputs consumed by all later stages.

Command (real Llama example, 10-sample smoke):

```bash
python run_model.py \
  --api_env_file ./api_values.env \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --data_path ./datasets \
  --dataset_name qasper.pkl \
  --dataset_size 10 \
  --save_path ./results \
  --save_name run_model_llama8b_qasper_10 \
  --model_judge gpt-4o-mini
```

Expected output:
- `results/run_model_llama8b_qasper_10.pkl`

## Section B: TruthTorchLM Baseline UQ (Step 2)

Purpose:
- Runs baseline uncertainty methods on step-1 output.

```bash
python run_uq.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --data_path ./results \
  --dataset_name run_model_llama8b_qasper_10.pkl \
  --save_path ./uq_results \
  --save_name run_uq_llama8b_qasper_10 \
  --number_of_generations 1 \
  --entailment_model microsoft/deberta-base-mnli \
  --entailment_device cuda:0
```

Expected output:
- `uq_results/run_uq_llama8b_qasper_10.pkl`

## Section C: Easy-Context Generation (Step 3)

Purpose:
- Creates context perturbation strings used by hidden-state extraction.

Command (real-run default style):

```bash
python easy_context.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --tensor_parallel_size 4 \
  --data_path ./datasets \
  --dataset_name qasper_train.pkl \
  --sample_num 1000 \
  --save_path ./easy_contexts \
  --save_name llama8b_qasper_train_easy_contexts.pkl
```

Expected output:
- `easy_contexts/llama8b_qasper_train_easy_contexts.pkl`

## Section D: Hidden-State Extraction (Step 4)

Run all four variants.

### D.1 Regular (base)

```bash
python extract_hidden_states.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --data_path ./results \
  --data_name run_model_llama8b_qasper_10.pkl \
  --save_path ./hidden_states \
  --save_name llama8b_smoke10 \
  --perturb_modes regular \
  --prompt regular \
  --sample_num 10 \
  --store_probs \
  --all_states
```

### D.2 Context

```bash
python extract_hidden_states.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --data_path ./results \
  --data_name run_model_llama8b_qasper_10.pkl \
  --save_path ./hidden_states \
  --save_name llama8b_smoke10 \
  --perturb_modes regular \
  --prompt context \
  --sample_num 10
```

### D.3 Honesty

```bash
python extract_hidden_states.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --data_path ./results \
  --data_name run_model_llama8b_qasper_10.pkl \
  --save_path ./hidden_states \
  --save_name llama8b_smoke10 \
  --perturb_modes regular \
  --prompt honesty \
  --sample_num 10
```

### D.4 Easy-context

```bash
python extract_hidden_states.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --data_path ./results \
  --data_name run_model_llama8b_qasper_10.pkl \
  --easy_context_path ./easy_contexts \
  --easy_context_name llama8b_qasper_train_easy_contexts.pkl \
  --save_path ./hidden_states \
  --save_name llama8b_smoke10 \
  --perturb_modes easy_context \
  --prompt regular \
  --sample_num 10
```

## Section E: Feature-Gaps (Step 5)

```bash
python run_feature_gaps.py \
  --hidden_states_path ./hidden_states \
  --train_files \
    "llama8b_smoke10_run_model_llama8b_qasper_10.pkl_regular_['easy_context']" \
    "llama8b_smoke10_run_model_llama8b_qasper_10.pkl_context_['regular']" \
    "llama8b_smoke10_run_model_llama8b_qasper_10.pkl_honesty_['regular']" \
  --val_files \
    "llama8b_smoke10_run_model_llama8b_qasper_10.pkl_regular_[]" \
    "llama8b_smoke10_run_model_llama8b_qasper_10.pkl_regular_[]" \
    "llama8b_smoke10_run_model_llama8b_qasper_10.pkl_regular_[]" \
  --test_files \
    "llama8b_smoke10_run_model_llama8b_qasper_10.pkl_regular_[]" \
    "llama8b_smoke10_run_model_llama8b_qasper_10.pkl_regular_[]" \
    "llama8b_smoke10_run_model_llama8b_qasper_10.pkl_regular_[]" \
  --train_num 10 \
  --val_num 10 \
  --test_num 10 \
  --output_path ./uq_results \
  --save_name feature_gaps_llama8b_qasper_10.pkl
```

## Section F: SAPLMA (Step 6)

```bash
python run_saplma.py \
  --hidden_states_path ./hidden_states \
  --train_file "llama8b_smoke10_run_model_llama8b_qasper_10.pkl_regular_[]" \
  --val_file "llama8b_smoke10_run_model_llama8b_qasper_10.pkl_regular_[]" \
  --test_file "llama8b_smoke10_run_model_llama8b_qasper_10.pkl_regular_[]" \
  --train_num 10 \
  --val_num 10 \
  --test_num 10 \
  --output_path ./uq_results \
  --save_name saplma_llama8b_qasper_10.pkl
```

## Section G: LookbackLens (Step 7)

```bash
python run_lookbacklens.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --results_path ./results \
  --train_results_file run_model_llama8b_qasper_10.pkl \
  --test_results_file run_model_llama8b_qasper_10.pkl \
  --train_num_samples 10 \
  --output_path ./uq_results \
  --save_name lookbacklens_llama8b_qasper_10.pkl
```

## Validation Status (This Machine)

I attempted the README order before finalizing this document:
- `run_model.py` starts correctly, but this runtime reports CUDA initialization errors and falls back to CPU, making generation+judge stages very slow.
- For reproducible paper runs, execute on a proper CUDA-enabled machine.

## Final Output Check

```bash
ls -lh ./results ./easy_contexts ./hidden_states ./uq_results
```
