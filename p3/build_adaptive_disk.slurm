#!/bin/bash
#SBATCH --job-name=p3_adaptive_index
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=p3/p3_adaptive_index_%j.out
#SBATCH --error=p3/p3_adaptive_index_%j.err

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
DATA="${DATA:-$REPO/sae-simlm-k32-baseline/data_titled}"
P3_OUT="${P3_OUT:-$DATA/p3_full_eval}"
CACHE="${CACHE:-$P3_OUT/top128_cache}"
THRESHOLD="${THRESHOLD:-0.49723}"
MIN_CONCEPTS="${MIN_CONCEPTS:-8}"
MAX_CONCEPTS="${MAX_CONCEPTS:-128}"
TAG="${TAG:-adaptive_t${THRESHOLD}}"
STATS="$P3_OUT/adaptive_exact_stats/threshold_$(printf '%0.5f' "$THRESHOLD")"
INDEX="$P3_OUT/${TAG}.diskindex"

cd "$REPO"
module load anaconda
eval "$(conda shell.bash hook)"
conda activate dl

export PYTHONPATH="$REPO:$REPO/upstream/Decoding-Dense-Embeddings/cl-sr${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$STATS/metadata.json" ]]; then
  python -u -m p3.compute_adaptive_corpus_stats \
    --cache "$CACHE" \
    --output "$(dirname "$STATS")" \
    --thresholds "$THRESHOLD" \
    --min-concepts "$MIN_CONCEPTS"
fi

python -u -m p3.build_adaptive_disk \
  --cache "$CACHE" --stats "$STATS" --output "$INDEX" \
  --threshold "$THRESHOLD" \
  --min-concepts "$MIN_CONCEPTS" --max-concepts "$MAX_CONCEPTS"

python -u -m p3.sort_adaptive_shards --index-dir "$INDEX"
