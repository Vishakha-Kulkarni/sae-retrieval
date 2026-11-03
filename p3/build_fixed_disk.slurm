#!/bin/bash
#SBATCH --job-name=fixed_disk_build
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=p3/fixed_disk_build_%j.out
#SBATCH --error=p3/fixed_disk_build_%j.err

set -eo pipefail

module load anaconda
eval "$(conda shell.bash hook)"
conda activate dl

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO"

export PYTHONPATH="$PWD:$PWD/upstream/Decoding-Dense-Embeddings/cl-sr${PYTHONPATH:+:$PYTHONPATH}"

P3_OUT="$PWD/sae-simlm-k32-baseline/data_titled/p3_full_eval"
CACHE="$P3_OUT/top128_cache"

K="${K:?Submit with K=32, 48, or 64}"

python -u p3/build_fixed_disk.py \
  --cache "$CACHE" \
  --output "$P3_OUT/fixed${K}.diskindex" \
  --k "$K"
