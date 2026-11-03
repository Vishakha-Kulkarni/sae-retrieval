#!/bin/bash
#SBATCH --job-name=p3_adaptive_eval
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=p3/p3_adaptive_eval_%j.out
#SBATCH --error=p3/p3_adaptive_eval_%j.err

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
DATA="${DATA:-$REPO/sae-simlm-k32-baseline/data_titled}"
P3_OUT="${P3_OUT:-$DATA/p3_full_eval}"
TAG="${TAG:-adaptive_t0.49723}"
INDEX="${INDEX:-$P3_OUT/${TAG}.diskindex}"
QUERIES="${QUERIES:-$P3_OUT/validation.query.json}"
QRELS="${QRELS:-$DATA/split/validation.qrels}"
CHECKPOINT="${CHECKPOINT:-$P3_OUT/${TAG}.validation.checkpoint.json}"
OUTPUT="${OUTPUT:-$P3_OUT/${TAG}.validation.json}"

cd "$REPO"
module load anaconda
eval "$(conda shell.bash hook)"
conda activate dl

export PYTHONPATH="$REPO:$REPO/upstream/Decoding-Dense-Embeddings/cl-sr${PYTHONPATH:+:$PYTHONPATH}"

python -u -m p3.evaluate_fixed_disk \
  --index "$INDEX" --queries "$QUERIES" --qrels "$QRELS" \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT" --every 5
