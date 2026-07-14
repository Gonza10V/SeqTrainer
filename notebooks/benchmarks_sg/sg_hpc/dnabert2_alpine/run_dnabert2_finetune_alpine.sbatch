#!/bin/bash
#SBATCH --job-name=seqtrainer-dnabert2-ft
#SBATCH --partition=aa100
#SBATCH --qos=gpu-long
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=seqtrainer-dnabert2-ft-%j.out
#SBATCH --error=seqtrainer-dnabert2-ft-%j.err

set -euo pipefail

if [[ -z "${BUNDLE_ROOT:-}" ]]; then
  echo "Set BUNDLE_ROOT to the staged offline bundle, for example /projects/$USER/seqtrainer-offline" >&2
  exit 2
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PIP_NO_INDEX=1
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

IMAGE="${BUNDLE_ROOT}/image/seqtrainer-alpine-gpu.sif"
REPO_DIR="${BUNDLE_ROOT}/repository/SeqTrainer"
SCRATCH_ROOT="${SCRATCH:-/scratch/alpine/${USER}}/seqtrainer-dnabert2-${SLURM_JOB_ID}"
FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR:-/projects/${USER}/seqtrainer-results/dnabert2/${SLURM_JOB_ID}}"
CONFIG_PATH="/repo/config-examples/benchmarks/dnabert2_finetune_alpine_offline.toml"
RUN_OUTPUT_DIR="/work/outputs/benchmarks/dnabert2_finetune_alpine_offline"

mkdir -p "${SCRATCH_ROOT}" "${FINAL_OUTPUT_DIR}"

echo "SeqTrainer DNABERT2 offline Alpine run"
echo "BUNDLE_ROOT=${BUNDLE_ROOT}"
echo "SCRATCH_ROOT=${SCRATCH_ROOT}"
echo "FINAL_OUTPUT_DIR=${FINAL_OUTPUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"

apptainer exec --nv --cleanenv \
  --bind "${BUNDLE_ROOT}:/bundle:ro" \
  --bind "${REPO_DIR}:/repo:ro" \
  --bind "${BUNDLE_ROOT}/models:/models:ro" \
  --bind "${BUNDLE_ROOT}/data:/data:ro" \
  --bind "${SCRATCH_ROOT}:/work" \
  "${IMAGE}" \
  python /repo/tools/offline_prep/validate_offline_bundle.py --bundle-root /bundle --model dnabert2

apptainer exec --nv --cleanenv \
  --bind "${REPO_DIR}:/repo:ro" \
  --bind "${BUNDLE_ROOT}/models:/models:ro" \
  --bind "${BUNDLE_ROOT}/data:/data:ro" \
  --bind "${SCRATCH_ROOT}:/work" \
  "${IMAGE}" \
  seqtrainer benchmark run "${CONFIG_PATH}" \
    --base-dir / \
    --output-dir "${RUN_OUTPUT_DIR}" \
    --strict

cp -a "${SCRATCH_ROOT}/outputs" "${FINAL_OUTPUT_DIR}/"
echo "Done. Results copied to ${FINAL_OUTPUT_DIR}"
