#!/bin/bash
#SBATCH --job-name=p3_top128
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=p3/p3_top128_%j.out
#SBATCH --error=p3/p3_top128_%j.err

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
DATA="${DATA:-$REPO/sae-simlm-k32-baseline/data_titled}"
SAE="${SAE:-$REPO/sae-simlm-k32-baseline/B0_final_atom.pt}"
P3_OUT="${P3_OUT:-$DATA/p3_full_eval}"
BATCH_SIZE="${BATCH_SIZE:-512}"

cd "$REPO"
module load anaconda
eval "$(conda shell.bash hook)"
conda activate dl

export PYTHONPATH="$REPO:$REPO/upstream/Decoding-Dense-Embeddings/cl-sr${PYTHONPATH:+:$PYTHONPATH}"

args=(
  --checkpoint "$SAE"
  --input-embeddings "$DATA/passages.npy"
  --output-dir "$P3_OUT/top128_cache"
  --batch-size "$BATCH_SIZE"
  --k 128
)
if [[ -f "$P3_OUT/top128_cache/checkpoint.json" ]]; then
  args+=(--resume)
fi

python -u -m p3.extract_topk_cache "${args[@]}"
