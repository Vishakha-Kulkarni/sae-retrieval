#!/usr/bin/env bash
#SBATCH --job-name=p1-index
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=01:30:00
#SBATCH --output=logs/p1-index-%j.out
#SBATCH --error=logs/p1-index-%j.err
set -euo pipefail
cd "${P1_ROOT:-$SLURM_SUBMIT_DIR}"
: "${P1_PYTHON:?Set P1_PYTHON}"
export PYTHONPATH="$PWD/runtime:$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$P1_PYTHON" -u -m p1.index --passages "$PWD/bundle/latents_B0_final_atom/passages.latent.jsonl" \
    --output-dir "$PWD/index-cap24" --max-concepts 24 --batch-docs 50000
