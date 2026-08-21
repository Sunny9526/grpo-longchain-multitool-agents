#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-agentrl}"

if [[ -f "${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}" ]]; then
    source "${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
else
    eval "$(conda shell.bash hook)"
fi
conda activate "${CONDA_ENV}"

# 注意: PyTorch 是 2.7.0+cu126(自带 CUDA 12.6 runtime),但系统 CUDA 工具链是 12.4
# CUDA_HOME 必须指向实际存在的系统目录,供 triton/deepspeed 找 ptxas
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-${CUDA_HOME}/bin/ptxas}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export DS_SKIP_TRITON=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VLLM_USE_V1=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENAI_API_KEY=dummy
export LITELLM_LOCAL_MODEL_COST_MAP="True"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export VLLM_LOGGING_LEVEL=ERROR

# expandable_segments disabled: incompatible with vLLM memory pool
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd "${PROJECT_ROOT}"
mkdir -p experiments/vanilla

# nohup python -m verl.trainer.main_ppo \
#     --config-path="${PROJECT_ROOT}/configs/train/grpo" \
#     --config-name=vanilla_grpo \
#     > experiments/vanilla/training.log 2>&1 &
# echo "Training PID: $!"

python -m verl.trainer.main_ppo \
    --config-path="${PROJECT_ROOT}/configs/train/grpo" \
    --config-name=vanilla_grpo