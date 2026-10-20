#!/bin/bash
# Smoke tier: the whole week-1 pipeline and a P5 B0 row on the first $SIZE MS MARCO passages, on a laptop.
# Numbers are not comparable to the paper or the real tier (small corpus, few SAE steps); they only show the
# pieces connect. The real tier runs the same commands with the shared $DATA/$CKPT paths from nscc/env.sh.
#
# RAW=~/msmarco SIZE=100000 scripts/smoke.sh
# RAW needs collection.tsv, queries.train.tsv, qrels.train.tsv, queries.dev.small.tsv, qrels.dev.small.tsv.
set -euo pipefail
RAW=${RAW:?set RAW to the MS MARCO download directory}
SIZE=${SIZE:-100000}
VAL_SIZE=${VAL_SIZE:-500}
STEPS=${STEPS:-2000}
OUT=${OUT:-smoke}
PY=${PY:-python}
CL_SR=upstream/Decoding-Dense-Embeddings/cl-sr

mkdir "$OUT"   # fails if a previous smoke run is still there
$PY -m p5.harness filter --qrels "$RAW/qrels.train.tsv" --queries "$RAW/queries.train.tsv" \
    --size "$SIZE" --prefix "$OUT/train_all"
$PY -m p5.harness filter --qrels "$RAW/qrels.dev.small.tsv" --queries "$RAW/queries.dev.small.tsv" \
    --size "$SIZE" --prefix "$OUT/dev"
$PY -m p1.prepare split --queries "$OUT/train_all.queries.tsv" --qrels "$OUT/train_all.qrels" \
    --output-dir "$OUT/split" --size "$VAL_SIZE"

$PY -m p2.encode passages --input "$RAW/collection.tsv" --output "$OUT/passages.npy" --limit "$SIZE"
$PY -m p2.encode queries --input "$OUT/split/train.queries.tsv" --output "$OUT/train_queries.npy"
$PY -m p2.encode queries --input "$OUT/split/validation.queries.tsv" --output "$OUT/val_queries.npy"
$PY -m p2.encode queries --input "$OUT/dev.queries.tsv" --output "$OUT/dev_queries.npy"
$PY -m p2.candidates --passages "$OUT/passages.npy" --queries "$OUT/train_queries.npy" \
    --qrels "$OUT/split/train.qrels" --output-prefix "$OUT/train"
$PY -m p2.candidates --passages "$OUT/passages.npy" --queries "$OUT/val_queries.npy" \
    --qrels "$OUT/split/validation.qrels" --output-prefix "$OUT/val"

# Smoke SAE: upstream objective, few steps; last.pt because best.pt depends on P2's NMSE gate.
$PY -m p2.train --data-dir "$OUT" --val-qrels "$OUT/split/validation.qrels" --output "$OUT/sae" \
    --rank-loss none --steps "$STEPS" --eval-every "$STEPS" --batch 1024
$PY -m p2.evaluate export --checkpoint "$OUT/sae/last.pt" --embeddings "$OUT/passages.npy" \
    --output "$OUT/passages.latents.jsonl"
for split in val dev; do
    $PY -m p2.evaluate export --checkpoint "$OUT/sae/last.pt" --embeddings "$OUT/${split}_queries.npy" \
        --output "$OUT/$split.query.json" --queries
done
$PY $CL_SR/indexing.py --passage_latents "$OUT/passages.latents.jsonl" --index_save_path "$OUT/index.pkl"

$PY -m p5.evaluate --name B0 --queries "$OUT/val.query.json" --index "$OUT/index.pkl" \
    --qrels "$OUT/split/validation.qrels" --run-output "$OUT/B0.val.run" --output "$OUT/B0.val.json"
