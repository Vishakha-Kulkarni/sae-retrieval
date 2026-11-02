#!/bin/bash
# Bring data_titled in line with docs/shared_artifacts.md (P5): dev-small dev queries (6,980),
# queries.dev.small.tsv, TREC DL 2019/2020 query embeddings + qrels, and MANIFEST checksums.
# Run on a login node (CPU, ~10 min, mostly hashing the 26 GB passage file):
#   bash scripts/nscc/finalize_data_titled.sh
set -eo pipefail
source "$(dirname "$0")/env.sh"   # conda activation is not nounset-safe, so -u comes after
set -u
D=${DATA_T:-$HOME/scratch/sae/data_titled}
TREC=$RAW/trec

echo "== dev-small queries"
python - "$D" "$RAW" <<'EOF'
import os, sys, numpy as np
D, RAW = sys.argv[1], sys.argv[2]
judged = {l.split()[0] for l in open(f'{RAW}/qrels.dev.small.tsv') if l.strip()}
ids = open(f'{D}/dev_queries.ids').read().split()
if len(ids) == len(judged) and set(ids) == judged:
    print('already dev-small:', len(ids))
    sys.exit()
keep = [i for i, q in enumerate(ids) if q in judged]
emb = np.ascontiguousarray(np.load(f'{D}/dev_queries.npy', mmap_mode='r')[keep], dtype=np.float32)
for f in ('dev_queries.npy', 'dev_queries.ids'):
    os.remove(f'{D}/{f}')  # were symlinks to the all-dev (101k) files in data/
np.save(f'{D}/dev_queries.npy', emb)
open(f'{D}/dev_queries.ids', 'w').write(''.join(ids[i] + '\n' for i in keep))
print('dev-small queries:', emb.shape)
EOF
awk 'NR==FNR{k[$1];next} ($1 in k)' "$RAW/qrels.dev.small.tsv" "$RAW/queries.dev.tsv" > "$RAW/queries.dev.small.tsv"
echo "queries.dev.small.tsv: $(wc -l < "$RAW/queries.dev.small.tsv") lines"

echo "== TREC DL 2019/2020"
mkdir -p "$TREC"
URL=https://msmarco.z22.web.core.windows.net/msmarcoranking
for y in 2019 2020; do
    n=dl${y: -2}
    [ -s "$TREC/msmarco-test$y-queries.tsv.gz" ] || wget -q -O "$TREC/msmarco-test$y-queries.tsv.gz" "$URL/msmarco-test$y-queries.tsv.gz"
    [ -s "$TREC/${y}qrels-pass.txt" ] || wget -q -O "$TREC/${y}qrels-pass.txt" "https://trec.nist.gov/data/deep/${y}qrels-pass.txt"
    [ -s "$D/${n}_queries.npy" ] || python -m p2.encode queries --input "$TREC/msmarco-test$y-queries.tsv.gz" \
        --output "$D/${n}_queries.npy" --device cpu
    cp "$TREC/${y}qrels-pass.txt" "$D/$n.qrels"
    echo "$n: $(wc -l < "$D/${n}_queries.ids") queries, $(cut -d' ' -f1 "$D/$n.qrels" | sort -u | wc -l) judged"
done

echo "== checksums"
(cd "$RAW" && sha256sum split/*) > "$D/MANIFEST.split.sha256"
(cd "$D" && sha256sum passages.npy ./*_queries.npy ./*_queries.ids ./*.cands.npy ./*.teacher.npy ./*.pos.npy \
    ./*.run ./*.qrels) > "$D/MANIFEST.sha256"
(cd "$D" && sha256sum -c --quiet MANIFEST.sha256) && echo "MANIFEST OK"
