#!/bin/bash
# Final evaluation of one P2 run (or B0) on MS MARCO dev-small and TREC DL 2019/2020:
#   Table 1 (NMSE, reconstructed dense retrieval), CL-SR (export -> fast CL-SR -> upstream evaluate.py),
#   and concept stability against B0. Submits all jobs; results land next to the checkpoint and in latents/<tag>/.
# bash scripts/nscc/evaluate_run.sh listwise_bm25_w0.1_s42          (uses <run>/best.pt)
# bash scripts/nscc/evaluate_run.sh B0 $HOME/scratch/sae/shared/B0_final_atom.pt
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/nscc/env.sh
TAG=${1:?run name, e.g. listwise_bm25_w0.1_s42}
CKPT_PATH=${2:-$CKPT/$TAG/best.pt}
[ -e "$CKPT_PATH" ] || { echo "no checkpoint $CKPT_PATH" >&2; exit 1; }
P="-P $PROJECT_ID"
echo "evaluating $CKPT_PATH as $TAG"
qsub $P -v "CKPT_PATH=$CKPT_PATH" scripts/nscc/table1.pbs
EXP=$(qsub $P -v "CKPT_PATH=$CKPT_PATH,TAG=$TAG" scripts/nscc/export_latents.pbs)
qsub $P -W depend=afterok:$EXP -v "TAG=$TAG,SPLITS=dev dl19 dl20" scripts/nscc/clsr_fast.pbs
[ "$CKPT_PATH" = "$BASELINE" ] || qsub $P -v "CKPT_PATH=$CKPT_PATH" scripts/nscc/concepts.pbs
echo "results: ${CKPT_PATH%.pt}.table1.json  $HOME/scratch/sae/latents/$TAG/clsr_fast_max24.{dev,dl19,dl20}.metrics.txt" \
     " ${CKPT_PATH%.pt}.concepts.json"
