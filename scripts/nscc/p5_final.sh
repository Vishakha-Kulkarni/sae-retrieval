#!/bin/bash
# Final one-shot P5 runs, after T and N are chosen on validation. Run on a login node from the repo:
#   T_ROW=T0.7 N_ROW=N8 bash scripts/nscc/p5_final.sh   # DL topics, B0/T/N/L on dev, DL19, DL20, Holm family on dev
#   SEEDS=1 bash scripts/nscc/p5_final.sh               # L for each extra seed $P5/expander_s<n>.pt, once the above ran
# Extra seeds train on validation like the first: qsub -v ROW=train,EXPANDER=$P5/expander_s1.pt,TRAIN_ARGS="--seed 1"
source scripts/nscc/env.sh
set -euo pipefail
sub() { qsub -P "$PROJECT_ID" "$@" scripts/nscc/p5.pbs; }
row() {  # split, row, param, dependencies
    sub ${4:+-W depend=afterok:$4} -v ROW=$2,SPLIT=$1,FINAL=1${3:+,PARAM=$3}
}

if [ -n "${SEEDS:-}" ]; then
    for expander in "$P5"/expander_s*.pt; do
        [ -e "$expander" ] || { echo "no $P5/expander_s*.pt yet" >&2; exit 1; }
        name=L_$(basename "$expander" .pt | sed 's/^expander_//')
        for split in dev dl19 dl20; do
            [ -e "$P5/B0.$split.run" ] || { echo "$P5/B0.$split.run missing: run the final rows first" >&2; exit 1; }
            sub -v ROW=L,SPLIT=$split,FINAL=1,NAME=$name,EXPANDER=$expander
        done
    done
    exit
fi

T_ROW=${T_ROW:?set T_ROW to the validation-chosen threshold row, e.g. T0.7}
N_ROW=${N_ROW:?set N_ROW to the validation-chosen neighbour row, e.g. N8}
for file in msmarco-test2019-queries.tsv msmarco-test2020-queries.tsv 2019qrels-pass.txt 2020qrels-pass.txt; do
    [ -e "$RAW/$file" ] || { echo "$RAW/$file missing: download the TREC DL topics and qrels first" >&2; exit 1; }
done

PF=$(sub -v ROW=prepare-final)
for split in dev dl19 dl20; do
    B0=$(row $split B0 "" "$PF")
    T=$(row $split T "${T_ROW#T}" "$PF:$B0")
    N=$(row $split N "${N_ROW#N}" "$PF:$B0")
    L=$(row $split L "" "$PF:$B0")
    [ $split = dev ] && DEV="$B0:$T:$N:$L"
done
sub -W depend=afterok:$DEV -v ROW=family,SPLIT=dev,FINAL=1,T_ROW=$T_ROW,N_ROW=$N_ROW
qstat -u "$USER"
