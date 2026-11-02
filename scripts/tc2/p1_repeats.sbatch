#!/usr/bin/env bash
#SBATCH --job-name=p1-seeds
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=05:30:00
#SBATCH --output=logs/p1-seeds-%j.out
#SBATCH --error=logs/p1-seeds-%j.err
set -euo pipefail
cd "${P1_ROOT:-$SLURM_SUBMIT_DIR}"
: "${P1_PYTHON:?Set P1_PYTHON}"
: "${P1_DATASET:?Set P1_DATASET to the prepared seed-42 data directory}"
export PYTHONPATH="$PWD/runtime:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=6
"$P1_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
# Reuse exactly the same training/validation queries, not seed-dependent samples.
for seed in 43 44; do
    output="$PWD/runs/repeat-s${seed}-j${SLURM_JOB_ID}"
    mkdir "$output"
    "$P1_PYTHON" -u -m p1.train --train "$P1_DATASET/train.jsonl" \
        --validation "$P1_DATASET/validation.jsonl" --stats "$P1_DATASET/stats.json" \
        --qrels "$P1_DATASET/validation.qrels" --output "$output/model.pt" \
        --epochs 10 --seed "$seed" --device cuda > "$output/training.jsonl"
    "$P1_PYTHON" -m p1.evaluate --data "$P1_DATASET/validation.jsonl" \
        --stats "$P1_DATASET/stats.json" --qrels "$P1_DATASET/validation.qrels" \
        --checkpoint "$output/model.pt" --run-output "$output/learned.run" > "$output/learned.metrics.json"
    echo "Completed seed $seed: $output"
done
