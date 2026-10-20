#!/bin/bash
# Smoke tier: the week-1 pipeline and every P5 validation row (B0, T, N, L) on the first $SIZE MS MARCO
# passages, on a laptop. Numbers are not comparable to the paper or the real tier (small corpus, few SAE
# steps); they only show the pieces connect. The P5 steps are scripts/p5.sh, the same script NSCC runs.
#
# RAW=~/msmarco SIZE=100000 scripts/smoke.sh
# RAW needs collection.tsv, queries.train.tsv and qrels.train.tsv.
set -euo pipefail
RAW=${RAW:?set RAW to the MS MARCO download directory}
SIZE=${SIZE:-100000}
VAL_SIZE=${VAL_SIZE:-500}
STEPS=${STEPS:-2000}
OUT=${OUT:-smoke}
PY=${PY:-python}

mkdir "$OUT"   # fails if a previous smoke run is still there
$PY -m p5.harness filter --qrels "$RAW/qrels.train.tsv" --queries "$RAW/queries.train.tsv" \
    --size "$SIZE" --prefix "$OUT/train_all"
$PY -m p1.prepare split --queries "$OUT/train_all.queries.tsv" --qrels "$OUT/train_all.qrels" \
    --output-dir "$OUT/split" --size "$VAL_SIZE"

$PY -m p2.encode passages --input "$RAW/collection.tsv" --output "$OUT/passages.npy" --limit "$SIZE"
$PY -m p2.encode queries --input "$OUT/split/train.queries.tsv" --output "$OUT/train_queries.npy"
$PY -m p2.encode queries --input "$OUT/split/validation.queries.tsv" --output "$OUT/val_queries.npy"
for split in train val; do
    qrels=$OUT/split/$([ $split = val ] && echo validation || echo train).qrels
    $PY -m p2.candidates --passages "$OUT/passages.npy" --queries "$OUT/${split}_queries.npy" \
        --qrels "$qrels" --output-prefix "$OUT/$split"
done

# Smoke SAE: upstream objective, few steps; last.pt because best.pt depends on P2's NMSE gate.
$PY -m p2.train --data-dir "$OUT" --val-qrels "$OUT/split/validation.qrels" --output "$OUT/sae" \
    --rank-loss none --steps "$STEPS" --eval-every "$STEPS" --batch 1024
$PY -m p2.evaluate export --checkpoint "$OUT/sae/last.pt" --embeddings "$OUT/passages.npy" \
    --output "$OUT/passages.latents.jsonl"
$PY upstream/Decoding-Dense-Embeddings/cl-sr/indexing.py --passage_latents "$OUT/passages.latents.jsonl" \
    --index_save_path "$OUT/index.pkl"
$PY -m p1.prepare corpus --passages "$OUT/passages.latents.jsonl" --database "$OUT/corpus.db" \
    --stats "$OUT/stats.json"

export PY RAW DATA=$OUT SAE=$OUT/sae/last.pt INDEX=$OUT/index.pkl STATS=$OUT/stats.json \
    CORPUS_DB=$OUT/corpus.db P5=$OUT/p5 VAL_QRELS=$OUT/split/validation.qrels MISMATCH=
ROW=prepare TRAIN_SAMPLE=${TRAIN_SAMPLE:-5000} bash scripts/p5.sh
ROW=train TRAIN_ARGS="--steps ${P5_STEPS:-300} --eval-every 100" bash scripts/p5.sh
ROW=B0 bash scripts/p5.sh
ROW=T PARAM=0.7 bash scripts/p5.sh
ROW=N PARAM=8 bash scripts/p5.sh
ROW=L bash scripts/p5.sh
