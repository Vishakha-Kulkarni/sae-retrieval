# Sourced by every P2 and P5 job. Edit these once for your NSCC account.
export PROJECT_ID=${PROJECT_ID:-personal-riddhima}          # your NSCC project code (qsub -P)
export REPO=${REPO:-$HOME/sae-retrieval}
export DATA=${DATA:-$HOME/scratch/sae/data}        # passages.npy, *_queries.npy, *.cands.npy ...
export RAW=${RAW:-$HOME/scratch/sae/raw}           # MS MARCO collection.tsv, queries, qrels
export CKPT=${CKPT:-$HOME/scratch/sae/p2}
# Baseline B0: upstream does not release a trained SAE, so B0 is trained with the upstream
# objective by baseline.pbs (or use P3's k=32 checkpoint). Its threshold is stored inside it.
export BASELINE=${BASELINE:-$HOME/scratch/sae/p2/B0_upstream_objective/last.pt}
# P5 (scripts/p5.sh). The shared CL-SR index (P4) and the stats/corpus built with the same concept cap
# (p1.prepare corpus --max-concepts = indexing.py --max_latents), all from the SAE in $SAE.
export SAE=${SAE:-$BASELINE}
export INDEX=${INDEX:-$HOME/scratch/sae/index/index.pkl}
export STATS=${STATS:-$HOME/scratch/sae/index/stats.json}
export CORPUS_DB=${CORPUS_DB:-$HOME/scratch/sae/index/corpus.db}
export P5=${P5:-$HOME/scratch/sae/p5}
export VAL_QRELS=${VAL_QRELS:-$RAW/split/validation.qrels}
export DEV_QRELS=${DEV_QRELS:-$RAW/qrels.dev.small.tsv}
export MISMATCH=${MISMATCH-$RAW/bm25/mismatch}     # p5.mismatch outputs from scripts/bm25.sh runs

module load miniforge3 2>/dev/null || module load anaconda3 2>/dev/null || true
source activate sae 2>/dev/null || conda activate sae 2>/dev/null || true
cd "$REPO"
