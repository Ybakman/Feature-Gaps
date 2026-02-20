#!/usr/bin/env bash
set -euo pipefail

# Fresh conda environment setup for Feature-Gaps.
# Usage:
#   bash setup_conda.sh
#   conda activate feature-gaps
#
# Optional:
#   ENV_NAME=myenv bash setup_conda.sh
#   WITH_CUDA=0 bash setup_conda.sh

ENV_NAME="${ENV_NAME:-feature-gaps}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
WITH_CUDA="${WITH_CUDA:-1}"          # 1 = install CUDA PyTorch wheels, 0 = CPU wheels
CUDA_WHL_INDEX="${CUDA_WHL_INDEX:-https://download.pytorch.org/whl/cu124}"

echo "[1/4] Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"

echo "[2/4] Activating env"
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "[3/4] Installing PyTorch"
python -m pip install --upgrade pip
if [[ "${WITH_CUDA}" == "1" ]]; then
  python -m pip install torch torchvision torchaudio --index-url "${CUDA_WHL_INDEX}"
else
  python -m pip install torch torchvision torchaudio
fi

echo "[4/4] Installing project dependencies"
python -m pip install \
  truthtorchlm \
  transformers \
  datasets \
  accelerate \
  sentencepiece \
  scikit-learn \
  tqdm \
  vllm

echo
echo "Setup complete."
echo "Next:"
echo "  conda activate ${ENV_NAME}"
