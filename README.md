# Feature Gaps Workflow (Single Guide)

This document is the single source of truth for running the pipeline.

Run order:
1. `run_model.py`
2. `run_uq.py`
3. `easy_context.py`
4. `extract_hidden_states.py`
5. `run_feature_gaps.py`
6. `run_saplma.py`
7. `run_lookbacklens.py`

The default values below use real run-style settings already used in this project (Mistral/Llama/Qasper-style setup).

## 0) Environment Setup

```bash
cd /home/yavuz/yavuz/feature_gaps
source /home/yavuz/miniconda3/etc/profile.d/conda.sh
conda activate /home/yavuz/miniconda3/envs/TruthTorchLLM
mkdir -p results uq_results hidden_states easy_contexts
```

For `run_model.py`, fill `api_values.env` first:

```env
OPENAI_API_KEY=...
GEMINI_API_KEY=...
GOOGLE_CLOUD_PROJECT=...
HF_HOME=/home/yavuz/yavuz/.cache/huggingface/
```

## 1) Run `run_model.py` First

Purpose:
- Generate model outputs and correctness labels over the dataset.
- This creates the base `.pkl` file consumed by `run_uq.py` and downstream methods.

Command (example 10-sample smoke run):

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

Output:
- `./results/run_model_qasper_10.pkl`

## 2) Run `run_uq.py`

Purpose:
- Run TruthTorchLM uncertainty methods over the `run_model.py` output.

Command:

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

Output:
- `./uq_results/run_uq_qasper_10.pkl`

## 3) Run `easy_context.py`

Purpose:
- Build easy-context perturbation text from `(question, ground_truths)`.
- This creates the file used by `extract_hidden_states.py --perturb_modes easy_context`.

Command (real-style default: Llama-3.1-8B + qasper_train):

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

Output:
- `./easy_contexts/llama8b_qasper_train_easy_contexts.pkl`

## 4) Run `extract_hidden_states.py`

Purpose:
- Create hidden-state artifacts for feature-gaps and SAPLMA runs.
- Run it multiple times for required perturbation settings.

### 4.1 Regular (base, with probs/all states)

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

### 4.2 Context perturbation

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

### 4.3 Honesty perturbation

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

### 4.4 Easy-context perturbation

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

## 5) Run `run_feature_gaps.py`

Purpose:
- Train/use feature-gaps method from hidden-state artifacts.

Command:

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

## 6) Run `run_saplma.py`

Purpose:
- Run SAPLMA on extracted hidden states.

Command:

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

## 7) Run `run_lookbacklens.py`

Purpose:
- Run LookbackLens using attention-based features.

Command:

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

## Final Check

```bash
ls -lh ./results ./easy_contexts ./hidden_states ./uq_results
```

