#!/usr/bin/env bash
# Submit from the isolated P1 workspace with P1_PYTHON set to an existing CUDA environment.
#SBATCH --job-name=p1-val5k
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=05:30:00
#SBATCH --output=logs/p1-%j.out
#SBATCH --error=logs/p1-%j.err
set -euo pipefail
ROOT=${P1_ROOT:-$SLURM_SUBMIT_DIR}
cd "$ROOT"
: "${P1_PYTHON:?Set P1_PYTHON to the CUDA-enabled Python executable}"
export PYTHONPATH="$ROOT/runtime:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export P1_BUNDLE=${P1_BUNDLE:-$ROOT/bundle}
export P1_OUTPUT=${P1_OUTPUT:-$ROOT/runs/train5000-val5000-s42-j$SLURM_JOB_ID}
export P1_TRAIN_LIMIT=${P1_TRAIN_LIMIT:-5000}
export P1_VAL_LIMIT=${P1_VAL_LIMIT:-5000}
export P1_EPOCHS=${P1_EPOCHS:-10}
export P1_SEED=${P1_SEED:-42}
export P1_DEVICE=cuda
"$P1_PYTHON" -u - <<'PY'
import hashlib
import os
from pathlib import Path
import torch

if not torch.cuda.is_available():
    raise RuntimeError('Slurm did not provide a usable CUDA GPU')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, flush=True)
revision = Path('SOURCE_COMMIT')
print('Source commit:', revision.read_text().strip() if revision.exists() else 'unrecorded', flush=True)
print('Slurm job:', os.environ.get('SLURM_JOB_ID'), flush=True)
print('Training/validation queries:', os.environ['P1_TRAIN_LIMIT'], os.environ['P1_VAL_LIMIT'], flush=True)
bundle = Path(os.environ['P1_BUNDLE'])
required = ['train.cands.npy', 'train.teacher.npy', 'val.cands.npy', 'val.teacher.npy',
            'train_queries.ids', 'val_queries.ids', 'val_queries.npy']
expected = {}
for line in (bundle / 'data_titled/MANIFEST.sha256').read_text().splitlines():
    digest, name = line.split(maxsplit=1)
    expected[name.removeprefix('./')] = digest
for name in required:
    path = bundle / 'data_titled' / name
    hasher = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            hasher.update(chunk)
    if hasher.hexdigest() != expected[name]:
        raise ValueError(f'Checksum mismatch after transfer: {path}')
    print('Verified:', name, flush=True)
PY
PY="$P1_PYTHON" bash scripts/p1_pilot.sh
