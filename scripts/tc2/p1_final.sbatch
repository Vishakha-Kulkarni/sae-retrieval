#!/usr/bin/env bash
#SBATCH --job-name=p1-final
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/p1-final-%j.out
#SBATCH --error=logs/p1-final-%j.err
set -euo pipefail
ROOT=${P1_ROOT:-$SLURM_SUBMIT_DIR}
export P1_ROOT="$ROOT"
cd "$ROOT/eval-src"
: "${P1_PYTHON:?Set P1_PYTHON}"
: "${P1_MODELS:?Set P1_MODELS to the fixed three-checkpoint JSON manifest}"
: "${P1_DATASET:?Set P1_DATASET to the original prepared training/validation dataset}"
export PYTHONPATH="$ROOT/eval-src:$ROOT/runtime${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
OUTPUT="$ROOT/runs/full-corpus-j$SLURM_JOB_ID"
mkdir "$OUTPUT"
"$P1_PYTHON" -u - <<'PY'
import json, math, os, tempfile
from pathlib import Path
import torch
from p1.index import build, PostingIndex
from p1.data import bm25
from p1.model import ImpactScorer, score_group
from p1.retrieve import Retriever
from p1.train import file_hash

assert torch.cuda.is_available()
print('Allocated GPU:',torch.cuda.get_device_name(0),flush=True)
print('Source commit:',Path('SOURCE_COMMIT').read_text().strip(),flush=True)
index_root=Path(os.environ['P1_ROOT'])/'index-cap24'
for line in (index_root/'SHA256SUMS').read_text().splitlines():
    digest,name=line.split(maxsplit=1)
    assert file_hash(index_root/name)==digest, 'Index checksum mismatch: '+name
print('Full index checksums verified',flush=True)
with tempfile.TemporaryDirectory(dir=Path(os.environ['P1_ROOT'])/'runs') as folder:
    root=Path(folder)
    docs=[{0:1.2,1:.5},{0:.8},{2:2.},{0:.4,2:.6},{3:.7}]
    source=root/'passages'
    source.write_text(''.join(json.dumps({'docid':i,'ids':list(d),'weight':list(d.values())})+'\n' for i,d in enumerate(docs)))
    build(source,root/'index',2,2)
    with PostingIndex(root/'index') as index:
        query={0:.7,1:.2}
        ranking,_=Retriever(index,device='cuda',chunk=1).search(query)
        for docid,score in ranking:
            assert math.isclose(score,bm25(query,docs[int(docid)],index.stats),abs_tol=1e-6)
        model=ImpactScorer()
        checkpoint=root/'model.pt'
        torch.save({'format_version':2,'hidden':32,'state_dict':model.state_dict(),'stats':index.stats},checkpoint)
        expected=score_group(model,{'query':query,'candidates':[{'concepts':d} for d in docs]},index.stats)
        ranking,_=Retriever(index,checkpoint,'cuda',chunk=1).search(query)
        for docid,score in ranking:
            assert math.isclose(score,expected[int(docid)].item(),abs_tol=2e-5)
print('GPU scoring parity checks passed',flush=True)
PY
PHASE=${P1_PHASE:-all}
case "$PHASE" in validation|heldout|all) ;; *) echo "Invalid P1_PHASE" >&2; exit 2;; esac
if [[ "$PHASE" != heldout ]]; then
    "$P1_PYTHON" -u -m p1.final validation --index "$ROOT/index-cap24" \
        --query-dir "$ROOT/final-queries" --bundle "$ROOT/bundle" --models "$P1_MODELS" \
        --output-dir "$OUTPUT/selection" --device cuda
fi
if [[ "$PHASE" != validation ]]; then
    # No held-out evaluation starts until the validation-only freeze file exists.
    FREEZE=${P1_FREEZE:-$OUTPUT/selection/freeze.json}
    "$P1_PYTHON" -u -m p1.final heldout --index "$ROOT/index-cap24" \
        --query-dir "$ROOT/final-queries" --bundle "$ROOT/bundle" \
        --freeze "$FREEZE" --train-ids "$P1_DATASET/train.qids" \
        --validation-ids "$P1_DATASET/validation.qids" --output-dir "$OUTPUT/heldout" --device cuda
    "$P1_PYTHON" -m p1.package --summary "$OUTPUT/heldout/summary.json" \
        --validation "$FREEZE" --output-dir "$OUTPUT/report"
fi
echo "Full P1 evaluation completed: $OUTPUT"
