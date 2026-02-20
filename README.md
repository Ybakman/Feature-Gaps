# Feature-Gaps

Official code for the paper experiments on uncertainty estimation in RAG-style QA using:
- baseline TruthTorchLM UQ methods,
- `feature_gaps`,
- SAPLMA,
- LookbackLens.

This repository is organized as a strict step-by-step pipeline.

## Pipeline Order

Run in this order:
1. `run_model.py`
2. `run_uq.py`
3. `easy_context.py`
4. `extract_hidden_states.py`
5. `run_feature_gaps.py`
6. `run_saplma.py`
7. `run_lookbacklens.py`

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

## Environment

```bash
cd /home/yavuz/yavuz/feature_gaps
source /home/yavuz/miniconda3/etc/profile.d/conda.sh
conda activate /home/yavuz/miniconda3/envs/TruthTorchLLM
mkdir -p results uq_results hidden_states easy_contexts
```

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

Command (paper-style default, 10-sample smoke):

```bash
python run_model.py \
  --api_env_file ./api_values.env \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --data_path ./datasets \
  --dataset_name qasper.pkl \
  --dataset_size 10 \
  --save_path ./results \
  --save_name run_model_qasper_10 \
  --model_judge gpt-4o-mini
```

Expected output:
- `results/run_model_qasper_10.pkl`

## Section B: TruthTorchLM Baseline UQ (Step 2)

Purpose:
- Runs baseline uncertainty methods on step-1 output.

```bash
python run_uq.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --data_path ./results \
  --dataset_name run_model_qasper_10.pkl \
  --save_path ./uq_results \
  --save_name run_uq_qasper_10 \
  --number_of_generations 1 \
  --entailment_model microsoft/deberta-base-mnli \
  --entailment_device cuda:0
```

Expected output:
- `uq_results/run_uq_qasper_10.pkl`

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
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --data_path ./results \
  --data_name run_model_qasper_10.pkl \
  --save_path ./hidden_states \
  --save_name qwen_smoke10 \
  --perturb_modes regular \
  --prompt regular \
  --sample_num 10 \
  --store_probs \
  --all_states
```

### D.2 Context

```bash
python extract_hidden_states.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --data_path ./results \
  --data_name run_model_qasper_10.pkl \
  --save_path ./hidden_states \
  --save_name qwen_smoke10 \
  --perturb_modes regular \
  --prompt context \
  --sample_num 10
```

### D.3 Honesty

```bash
python extract_hidden_states.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --data_path ./results \
  --data_name run_model_qasper_10.pkl \
  --save_path ./hidden_states \
  --save_name qwen_smoke10 \
  --perturb_modes regular \
  --prompt honesty \
  --sample_num 10
```

### D.4 Easy-context

```bash
python extract_hidden_states.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --data_path ./results \
  --data_name run_model_qasper_10.pkl \
  --easy_context_path ./easy_contexts \
  --easy_context_name llama8b_qasper_train_easy_contexts.pkl \
  --save_path ./hidden_states \
  --save_name qwen_smoke10 \
  --perturb_modes easy_context \
  --prompt regular \
  --sample_num 10
```

## Section E: Feature-Gaps (Step 5)

```bash
python run_feature_gaps.py \
  --hidden_states_path ./hidden_states \
  --train_files \
    "qwen_smoke10_run_model_qasper_10.pkl_regular_['easy_context']" \
    "qwen_smoke10_run_model_qasper_10.pkl_context_['regular']" \
    "qwen_smoke10_run_model_qasper_10.pkl_honesty_['regular']" \
  --val_files \
    "qwen_smoke10_run_model_qasper_10.pkl_regular_[]" \
    "qwen_smoke10_run_model_qasper_10.pkl_regular_[]" \
    "qwen_smoke10_run_model_qasper_10.pkl_regular_[]" \
  --test_files \
    "qwen_smoke10_run_model_qasper_10.pkl_regular_[]" \
    "qwen_smoke10_run_model_qasper_10.pkl_regular_[]" \
    "qwen_smoke10_run_model_qasper_10.pkl_regular_[]" \
  --train_num 10 \
  --val_num 10 \
  --test_num 10 \
  --output_path ./uq_results \
  --save_name feature_gaps_qasper_10.pkl
```

## Section F: SAPLMA (Step 6)

```bash
python run_saplma.py \
  --hidden_states_path ./hidden_states \
  --train_file "qwen_smoke10_run_model_qasper_10.pkl_regular_[]" \
  --val_file "qwen_smoke10_run_model_qasper_10.pkl_regular_[]" \
  --test_file "qwen_smoke10_run_model_qasper_10.pkl_regular_[]" \
  --train_num 10 \
  --val_num 10 \
  --test_num 10 \
  --output_path ./uq_results \
  --save_name saplma_qasper_10.pkl
```

## Section G: LookbackLens (Step 7)

```bash
python run_lookbacklens.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --results_path ./results \
  --train_results_file run_model_qasper_10.pkl \
  --test_results_file run_model_qasper_10.pkl \
  --train_num_samples 10 \
  --output_path ./uq_results \
  --save_name lookbacklens_qasper_10.pkl
```

## Validation Status (This Machine)

I attempted the README order before finalizing this document:
- `run_model.py` starts correctly, but this runtime reports CUDA initialization errors and falls back to CPU, making generation+judge stages very slow.
- For reproducible paper runs, execute on a proper CUDA-enabled machine.

## Final Output Check

```bash
ls -lh ./results ./easy_contexts ./hidden_states ./uq_results
```
