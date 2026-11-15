#!/bin/bash
# One P5 step. The same commands run on the smoke tier (scripts/smoke.sh) and on NSCC (scripts/nscc/p5.pbs);
# only the paths differ. Paths come from the environment (scripts/nscc/env.sh on NSCC):
#   DATA       p2.encode / p2.candidates outputs (*_queries.npy with .ids, train.run, val.run)
#   SAE        SAE checkpoint (B0) the index was built from; INDEX, STATS, CORPUS_DB describe that index
#   P5         output directory; VAL_QRELS, DEV_QRELS; RAW holds the TREC DL topics and qrels
#   MISMATCH   p5.mismatch output directory (validation.* and dev.*); empty skips the subset views
#
# ROW=prepare                 query latents for train/val and the expander's training groups
# ROW=prepare-final           query latents for dev, DL19, DL20 (DL topics are encoded here)
# ROW=train                   learned expander, selected on validation (TRAIN_ARGS passes extra flags)
# ROW=B0|T|N|L [SPLIT=val]    one ablation row; run B0 first, the other rows test against B0's run
#   T: PARAM = threshold scale; N: PARAM = budget; L: PARAM overrides the validation-chosen budget
# ROW=family T_ROW=T0.7 N_ROW=N8 [SPLIT=val|dev]   Holm-corrected L vs B0/T/N on the primary metrics
# SPLIT=dev|dl19|dl20 are the final one-shot runs and need FINAL=1 (after the 3 Nov freeze).
set -euo pipefail
ROW=${ROW:?set ROW}
SPLIT=${SPLIT:-val}
PY=${PY:-python}
EXPANDER=${EXPANDER:-$P5/expander.pt}
mkdir -p "$P5"

latents() {
    $PY -m p2.evaluate export --checkpoint "$SAE" --embeddings "$DATA/${1}_queries.npy" \
        --output "$P5/$1.query.json" --queries
}

case $ROW in
prepare)
    latents train
    latents val
    $PY - "$DATA/train_queries.ids" "${TRAIN_SAMPLE:-100000}" > "$P5/train.sample.qids" <<'EOF'
import random, sys
from pathlib import Path
qids = Path(sys.argv[1]).read_text().split()
random.Random(42).shuffle(qids)
print('\n'.join(sorted(qids[:int(sys.argv[2])])))
EOF
    $PY -m p1.prepare groups --queries "$P5/train.query.json" --database "$CORPUS_DB" \
        --teacher-run "$DATA/train.run" --qids "$P5/train.sample.qids" --output "$P5/train.groups.jsonl"
    $PY -m p1.prepare groups --queries "$P5/val.query.json" --database "$CORPUS_DB" \
        --teacher-run "$DATA/val.run" --output "$P5/val.groups.jsonl"
    exit ;;
prepare-final)
    for year in 2019 2020; do
        $PY -m p2.encode queries --input "$RAW/msmarco-test$year-queries.tsv" --output "$DATA/dl${year#20}_queries.npy"
    done
    for split in dev dl19 dl20; do latents $split; done
    exit ;;
train)
    # shellcheck disable=SC2086
    $PY -m p5.train --train-groups "$P5/train.groups.jsonl" --train-embeddings "$DATA/train_queries.npy" \
        --val-groups "$P5/val.groups.jsonl" --val-embeddings "$DATA/val_queries.npy" --val-qrels "$VAL_QRELS" \
        --checkpoint "$SAE" --stats "$STATS" --output "$EXPANDER" ${TRAIN_ARGS:-}
    exit ;;
family)
    # P5's formal claim: L against B0, T and N on the primary metrics, Holm-corrected. Needs MISMATCH and the
    # chosen rows' runs, e.g. T_ROW=T0.7 N_ROW=N8 (L_ROW defaults to L).
    if [ "$SPLIT" != val ] && [ "${FINAL:-0}" != 1 ]; then
        echo "SPLIT=$SPLIT is a final test split: set FINAL=1 after the freeze" >&2
        exit 2
    fi
    SUBSET=$([ "$SPLIT" = val ] && echo validation || echo "$SPLIT")
    QRELS=$([ "$SPLIT" = val ] && echo "$VAL_QRELS" || echo "$DEV_QRELS")
    L_ROW=${L_ROW:-L}
    $PY -m p5.harness family --qrels "$QRELS" --subset "${MISMATCH:?family needs MISMATCH}/$SUBSET.mismatch_1000.qids" \
        --run "$L_ROW=$P5/$L_ROW.$SPLIT.run" --against "B0=$P5/B0.$SPLIT.run" \
        "${T_ROW:?set T_ROW, e.g. T0.7}=$P5/$T_ROW.$SPLIT.run" "${N_ROW:?set N_ROW, e.g. N8}=$P5/$N_ROW.$SPLIT.run" \
        --output "$P5/family.$SPLIT.json"
    exit ;;
B0|T|N|L) ;;
*) echo "unknown ROW=$ROW" >&2; exit 2 ;;
esac

if [ "$SPLIT" != val ] && [ "${FINAL:-0}" != 1 ]; then
    echo "SPLIT=$SPLIT is a final test split: tune on val, then set FINAL=1 after the freeze" >&2
    exit 2
fi
case $SPLIT in
val) QRELS=$VAL_QRELS; SUBSET=validation ;;
dev) QRELS=$DEV_QRELS; SUBSET=dev ;;
dl19) QRELS=$RAW/2019qrels-pass.txt; SUBSET= ;;
dl20) QRELS=$RAW/2020qrels-pass.txt; SUBSET= ;;
*) echo "unknown SPLIT=$SPLIT" >&2; exit 2 ;;
esac

NAME=${NAME:-$ROW${PARAM:-}}  # NAME=L_s1 EXPANDER=… keeps extra training seeds apart
ROW_QUERIES=$P5/$NAME.$SPLIT.query.json
case $ROW in
B0) ROW_QUERIES=$P5/$SPLIT.query.json ;;
T) $PY -m p5.expand threshold --checkpoint "$SAE" --embeddings "$DATA/${SPLIT}_queries.npy" \
       --scale "${PARAM:?T needs PARAM=threshold scale}" --output "$ROW_QUERIES" ;;
N) $PY -m p5.expand neighbours --queries "$P5/$SPLIT.query.json" --checkpoint "$SAE" --stats "$STATS" \
       --budget "${PARAM:?N needs PARAM=budget}" --output "$ROW_QUERIES" ;;
L) $PY -m p5.expand learned --queries "$P5/$SPLIT.query.json" --embeddings "$DATA/${SPLIT}_queries.npy" \
       --checkpoint "$SAE" --stats "$STATS" --expander "$EXPANDER" ${PARAM:+--budget "$PARAM"} \
       --output "$ROW_QUERIES" ;;
esac

EXTRA=()
if [ -n "${MISMATCH:-}" ] && [ -n "$SUBSET" ]; then
    for K in 10 100 1000; do EXTRA+=(--subset "mismatch_$K=$MISMATCH/$SUBSET.mismatch_$K.qids"); done
fi
[ "$ROW" = B0 ] || EXTRA+=(--baseline-run "$P5/B0.$SPLIT.run")
# Linear NDCG gain matches trec_eval on graded TREC DL qrels; binary MS MARCO qrels are unaffected.
$PY -m p5.evaluate --name "$NAME" --queries "$ROW_QUERIES" --index "$INDEX" --qrels "$QRELS" \
    ${EXTRA[@]+"${EXTRA[@]}"} --gain linear --check-upstream "${UPSTREAM_CHECKS:-2}" \
    --run-output "$P5/$NAME.$SPLIT.run" --output "$P5/$NAME.$SPLIT.json"
