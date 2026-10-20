# Sourced by every P2 job. Edit these once for your NSCC account.
export PROJECT_ID=${PROJECT_ID:-personal-riddhima}          # your NSCC project code (qsub -P)
export REPO=${REPO:-$HOME/sae-retrieval}
export DATA=${DATA:-$HOME/scratch/sae/data}        # passages.npy, *_queries.npy, *.cands.npy ...
export RAW=${RAW:-$HOME/scratch/sae/raw}           # MS MARCO collection.tsv, queries, qrels
export CKPT=${CKPT:-$HOME/scratch/sae/p2}
# Baseline B0: upstream does not release a trained SAE, so B0 is trained with the upstream
# objective by baseline.pbs (or use P3's k=32 checkpoint). Its threshold is stored inside it.
export BASELINE=${BASELINE:-$HOME/scratch/sae/p2/B0_upstream_objective/last.pt}

module load miniforge3 2>/dev/null || module load anaconda3 2>/dev/null || true
source activate sae 2>/dev/null || conda activate sae 2>/dev/null || true
cd "$REPO"
