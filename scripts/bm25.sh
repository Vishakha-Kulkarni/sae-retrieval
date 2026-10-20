#!/bin/bash
# BM25 top-1000 runs for the vocabulary-mismatch subsets (P5 harness, plan KTD3).
# Pyserini needs Python 3.10 and Java 21, so it lives in its own environment; only run files leave it:
#   conda create -n pyserini python=3.10 openjdk=21 -c conda-forge && conda activate pyserini
#   pip install pyserini
# The prebuilt msmarco-v1-passage index downloads once to ~/.cache/pyserini/indexes/.
#
# VALIDATION_TOPICS=$RAW/split/validation.queries.tsv OUT=$RAW/bm25 bash scripts/bm25.sh
# Default BM25 is k1=0.9, b=0.4. If the dev K=1000 mismatch count is far from the paper's 988,
# rerun with K1=0.82 B=0.68 (Pyserini's tuned setting) and keep whichever matches.
set -euo pipefail
OUT=${OUT:?set OUT to the output directory}
K1=${K1:-0.9}
B=${B:-0.4}
THREADS=${THREADS:-8}
mkdir -p "$OUT"
tag="k${K1}_b${B}"

run() {  # topics, output; TREC 6-column output (no --output-format msmarco: p5 reads 4 or 6 columns)
    python -m pyserini.search.lucene --index msmarco-v1-passage --topics "$1" --output "$2" \
        --hits 1000 --bm25 --k1 "$K1" --b "$B" --threads "$THREADS" --batch-size 64
}

run msmarco-passage-dev-subset "$OUT/bm25.dev.$tag.trec"
if [ -n "${VALIDATION_TOPICS:-}" ]; then
    run "$VALIDATION_TOPICS" "$OUT/bm25.validation.$tag.trec"
fi
echo "Next (main environment): python -m p5.mismatch --run $OUT/bm25.dev.$tag.trec --qrels <dev qrels> --output-dir <dir> --name dev"
