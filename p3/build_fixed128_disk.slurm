#!/bin/bash
#SBATCH --job-name=fixed128_build
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=p3/fixed128_build_%j.out
#SBATCH --error=p3/fixed128_build_%j.err

set -eo pipefail

module load anaconda
eval "$(conda shell.bash hook)"
conda activate dl

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO"

export PYTHONPATH="$PWD/upstream/Decoding-Dense-Embeddings/cl-sr:$PWD${PYTHONPATH:+:$PYTHONPATH}"

P3_OUT="$PWD/sae-simlm-k32-baseline/data_titled/p3_full_eval"

python -u p3/build_fixed128_disk.py \
  --cache "$P3_OUT/top128_cache" \
  --output "$P3_OUT/fixed128.diskindex" \
  --chunk-docs 50000
