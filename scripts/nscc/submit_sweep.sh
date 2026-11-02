#!/bin/bash
# Final P2 ablation, all fine-tuned from the frozen B0 (env.sh BASELINE) with the same budget:
# 20k steps, lr 1e-5, per-query standardised scores. best.pt in each run = best validation CL-SR
# among checkpoints with NMSE <= 1.10 x B0.
# PROJECT_ID=personal-riddhima bash scripts/nscc/submit_sweep.sh     (SEED=43 bash ... for extra seeds)
set -euo pipefail
cd "$(dirname "$0")/../.."
COMMON="STEPS=${STEPS:-20000},LR=${LR:-1e-5},SEED=${SEED:-42}"
submit() { qsub -P "$PROJECT_ID" -v "NAME=$1,LOSS=$2,SPACE=$3,WEIGHT=$4,$COMMON" -N "p2_$1" scripts/nscc/train.pbs; }
submit B1_recon_only none recon 0         # control: same steps, reconstruction only
submit naacl_w0.1 naacl recon 0.1          # Kang et al. NAACL 2025 objective
submit listwise_recon_w0.1 listwise recon 0.1
submit listwise_latent_w0.1 listwise latent 0.1
for w in 0.05 0.1 0.2; do                  # ours: differentiable CL-SR (BM25) space
  submit listwise_bm25_w$w listwise bm25 $w
done
