#!/bin/bash
# Submit the full P2 ablation: control, NAACL baseline, and ours (two score spaces x three weights).
# PROJECT_ID=xxxx bash scripts/nscc/submit_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
submit() { qsub -P "$PROJECT_ID" -v "PROJECT_ID=$PROJECT_ID,NAME=$1,LOSS=$2,SPACE=$3,WEIGHT=$4,SEED=${SEED:-42}" \
           -N "p2_$1" scripts/nscc/train.pbs; }
# Control: same fine-tuning steps with reconstruction only, so gains are not just "trained longer".
submit recon_only none recon 0
for w in 0.1 0.5 1.0; do
  submit naacl_w$w naacl recon $w
  submit listwise_recon_w$w listwise recon $w
  submit listwise_latent_w$w listwise latent $w
done
