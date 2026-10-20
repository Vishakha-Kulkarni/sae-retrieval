#!/bin/bash
# The shared 5,000-query validation split, on any machine, in under a minute (no GPU, no Java).
#
# OUT=~/msmarco_raw bash scripts/split.sh
#
# Downloads from the official MS MARCO site (about 30 MB) into $OUT:
#   queries.train.tsv, qrels.train.tsv, queries.dev.tsv, queries.eval.tsv, qrels.dev.small.tsv
# Writes:
#   $OUT/queries.dev.small.tsv   the 6,980 dev queries that have qrels.dev.small.tsv labels
#   $OUT/split/validation.{qids,queries.tsv,qrels}   5,000 train queries for tuning
#   $OUT/split/train.{qids,queries.tsv,qrels}        the other 497,939 labelled train queries
# and checks every file against splits/SHA256SUMS, so the split is identical for everyone. Point $RAW at
# $OUT (or copy $OUT/split to $RAW/split). Embeddings and candidate lists need scripts/nscc/shared.pbs.
# Existing files are reused, so rerunning only checks them.
set -euo pipefail
OUT=${OUT:?set OUT to a directory for the MS MARCO files, e.g. OUT=~/msmarco_raw}
PY=${PY:-python}
REPO=$(cd "$(dirname "$0")/.." && pwd)
URL=https://msmarco.z22.web.core.windows.net/msmarcoranking
mkdir -p "$OUT"

for file in queries.tar.gz qrels.train.tsv qrels.dev.small.tsv; do
    [ -e "$OUT/$file" ] || curl -fSL --retry 3 -o "$OUT/$file" "$URL/$file"
done
[ -e "$OUT/queries.train.tsv" ] || tar -xzf "$OUT/queries.tar.gz" -C "$OUT"
if [ ! -e "$OUT/queries.dev.small.tsv" ]; then  # not shipped in queries.tar.gz
    awk -F'\t' 'NR==FNR{keep[$1]=1; next} ($1 in keep)' "$OUT/qrels.dev.small.tsv" "$OUT/queries.dev.tsv" \
        > "$OUT/queries.dev.small.tsv"
fi
if [ ! -e "$OUT/split" ]; then
    (cd "$REPO" && $PY -m p1.prepare split --queries "$OUT/queries.train.tsv" --qrels "$OUT/qrels.train.tsv" \
        --output-dir "$OUT/split" --size 5000 --seed 42)
fi

if command -v sha256sum >/dev/null; then check="sha256sum -c"; else check="shasum -a 256 -c"; fi
if (cd "$OUT" && $check --quiet "$REPO/splits/SHA256SUMS"); then
    echo "OK: $OUT/split matches splits/SHA256SUMS (5,000 validation, 497,939 train queries)"
else
    echo "MISMATCH: files above differ from splits/SHA256SUMS; do not tune on this split" >&2
    exit 1
fi
