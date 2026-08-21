#!/usr/bin/env bash
# Usage:
#   bash setup.sh
#   DOWNLOAD_MODELS=1 bash setup.sh
#   DOWNLOAD_MODELS=1 DOWNLOAD_72B=1 bash setup.sh

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-agentrl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found in PATH." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
cd "${ROOT_DIR}"

echo "=== [1/5] Create conda environment: ${ENV_NAME} ==="
if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "Environment ${ENV_NAME} already exists."
else
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"

echo "=== [2/5] Install PyTorch 2.7 (CUDA 12.6 wheel) ==="
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126

echo "=== [3/5] Install project dependencies ==="
python -m pip install -r "${ROOT_DIR}/requirements.txt"

echo "=== [4/5] Install vendored veRL and tau-bench ==="
python -m pip install -e "${ROOT_DIR}/verl"
python -m pip install -e "${ROOT_DIR}/tau-bench"

echo "=== [5/5] Optional model download ==="
if [[ "${DOWNLOAD_MODELS:-0}" == "1" ]]; then
    python -m pip install --upgrade modelscope
    ROOT_DIR="${ROOT_DIR}" DOWNLOAD_72B="${DOWNLOAD_72B:-0}" python - <<'PY'
import os
from pathlib import Path

from modelscope import snapshot_download

model_dir = Path(os.environ["ROOT_DIR"]) / "models"
snapshot_download("Qwen/Qwen2.5-7B-Instruct", cache_dir=str(model_dir))
if os.environ["DOWNLOAD_72B"] == "1":
    snapshot_download("Qwen/Qwen2.5-72B-Instruct-AWQ", cache_dir=str(model_dir))
PY
else
    echo "Skipped. Set DOWNLOAD_MODELS=1 to download the 7B model."
fi

echo "Setup complete. Activate with: conda activate ${ENV_NAME}"