#!/bin/bash
#SBATCH --job-name=seqtrainer-ipromp
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=seqtrainer-ipromp-%j.out
#SBATCH --error=seqtrainer-ipromp-%j.err

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
SCRATCH_ROOT="${SCRATCH:-/scratch/alpine/${USER}}/seqtrainer-ipromp-${SLURM_JOB_ID}"
FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR:-/projects/${USER}/seqtrainer-results/ipromp/${SLURM_JOB_ID}}"
CONFIG_PATH="/repo/config-examples/benchmarks/ipromp_external_alpine_offline.toml"
RUN_OUTPUT_DIR="/work/outputs/benchmarks/ipromp_external_alpine_offline"

mkdir -p "${SCRATCH_ROOT}/external_predictions" "${FINAL_OUTPUT_DIR}"

echo "SeqTrainer iPro-MP offline Alpine run"
echo "BUNDLE_ROOT=${BUNDLE_ROOT}"
echo "SCRATCH_ROOT=${SCRATCH_ROOT}"
echo "FINAL_OUTPUT_DIR=${FINAL_OUTPUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"

COMMON_BINDS=(
  --bind "${REPO_DIR}:/repo:ro"
  --bind "${BUNDLE_ROOT}/models:/models:ro"
  --bind "${BUNDLE_ROOT}/data:/data:ro"
  --bind "${SCRATCH_ROOT}:/work"
)

apptainer exec --nv --cleanenv \
  --bind "${BUNDLE_ROOT}:/bundle:ro" \
  "${COMMON_BINDS[@]}" \
  "${IMAGE}" \
  python /repo/tools/offline_prep/validate_offline_bundle.py --bundle-root /bundle --model ipromp

apptainer exec --nv --cleanenv \
  "${COMMON_BINDS[@]}" \
  "${IMAGE}" \
  seqtrainer benchmark prepare-ipromp "${CONFIG_PATH}" \
    --base-dir / \
    --output-dir "${RUN_OUTPUT_DIR}"

for split in train validation test; do
  apptainer exec --nv --cleanenv \
    "${COMMON_BINDS[@]}" \
    "${IMAGE}" \
    python -m seqtrainer.adapters.ipromp_offline \
      --input-fasta "${RUN_OUTPUT_DIR}/ipromp_fasta/${split}.fasta" \
      --output-csv "/work/external_predictions/${split}_predictions.csv" \
      --metadata-json "/work/external_predictions/${split}_metadata.json" \
      --dnabert-dir /models/DNABERT-6 \
      --model-dir /models/ipromp_ecoli \
      --species-id 10 \
      --token-max-length 128 \
      --batch-size 16 \
      --device cuda
done

apptainer exec --nv --cleanenv \
  "${COMMON_BINDS[@]}" \
  "${IMAGE}" \
  seqtrainer benchmark run "${CONFIG_PATH}" \
    --base-dir / \
    --output-dir "${RUN_OUTPUT_DIR}" \
    --strict

cp -a "${SCRATCH_ROOT}/outputs" "${FINAL_OUTPUT_DIR}/"
cp -a "${SCRATCH_ROOT}/external_predictions" "${FINAL_OUTPUT_DIR}/"
echo "Done. Results copied to ${FINAL_OUTPUT_DIR}"
