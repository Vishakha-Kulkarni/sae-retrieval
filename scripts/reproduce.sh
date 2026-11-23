#!/bin/bash
# Reproduce every result in the report, stage by stage.
#
#   bash scripts/reproduce.sh <stage> [<stage> ...]
#   DRY_RUN=1 bash scripts/reproduce.sh all        # print every command without running it
#
# Stages, in dependency order (`all` runs them in this order):
#   setup     install requirements, fetch the upstream submodule
#   test      unit tests (CPU, seconds; no data needed)
#   smoke     B0 pipeline + query-expansion rows on a 100k-passage slice (laptop; numbers are not comparable)
#   data      MS MARCO download, shared split, SimLM embeddings, dense top-100 candidates  [GPU]
#   mismatch  BM25 vocabulary-mismatch subsets (needs Pyserini in $BM25_PY; optional: the
#             subsets are already committed under splits/p5_mismatch_k0.9_b0.4)
#   b0        reproduced baseline: SAE training, Table 2 rows, latents, CL-SR dev/DL19/DL20  [GPU]
#   scoring   learned impact scoring, tuned-BM25 baseline, 3 seeds                             [GPU]
#   sae       ranking-aware SAE: validation sweep, controls, 3 seeds of the selected run       [GPU]
#   sparsity  adaptive passage sparsity vs fixed k (validation only, as in the report)         [GPU]
#   pruning   concept pruning: learned gate vs DF on a Pyserini/Lucene index (needs Java 21)
#   expansion query concept expansion: L vs B0, T and N, Holm family, 3 seeds                  [GPU]
#
# Each stage is a sequence of idempotent steps: a step whose output already exists is skipped, so a
# stage can be rerun after a timeout. On NSCC/Slurm, wrap one stage in a job, e.g.
#   qsub -l select=1:ncpus=16:ngpus=1:mem=110gb -l walltime=24:00:00 -- /bin/bash -lc \
#        "cd $PWD && WORK=... bash scripts/reproduce.sh b0"
# Walltime per stage is roughly: data 12 h, b0 24 h (SAE) + 4 h, scoring 6 h, sae 6 x 8 h (+2 seeds),
# sparsity 12 h, pruning 8 h, expansion 12 h.
#
# Settings (environment variables, defaults in brackets):
#   WORK      output root for everything this script writes [./repro]
#   RAW       MS MARCO downloads and the split [$WORK/raw]
#   DATA      embeddings, candidates, teacher scores [$WORK/data_titled]
#   PY        Python with requirements.txt installed [python]
#   BM25_PY   Python with Pyserini (Python >= 3.12, Java 21) for `mismatch` and `pruning` [$PY]
#   DEVICE    [cuda]
#   B0_STEPS  SAE training steps for B0 [228000, as in the report]
#   SAE_SEL   ranking-aware SAE run selected on validation [listwise_bm25_w0.2, the report's choice]
#   T_ROW, N_ROW   query-expansion controls selected on validation [T0.5, N4, the report's choice]
#
# Selection is done on the 5,000-query validation split only. Stages that touch dev or TREC DL do so
# once, after the validation choice above is fixed. To follow the protocol strictly, run the
# validation part, read its outputs, and set SAE_SEL / T_ROW / N_ROW from them before the final part.
#
# Not scripted: the "Combined" rows of Table 3 (single seed-42 runs stacking methods on the
# ranking-aware SAE), and the k2 = 15 recalibration check, which the README explains how to run by hand.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
WORK=$(mkdir -p "${WORK:-$REPO/repro}" && cd "${WORK:-$REPO/repro}" && pwd)
RAW=${RAW:-$WORK/raw}
DATA=${DATA:-$WORK/data_titled}
PY=${PY:-python}
BM25_PY=${BM25_PY:-$PY}
DEVICE=${DEVICE:-cuda}
B0_STEPS=${B0_STEPS:-228000}
SAE_SEL=${SAE_SEL:-listwise_bm25_w0.2}
T_ROW=${T_ROW:-T0.5}
N_ROW=${N_ROW:-N4}
SEEDS=(42 43 44)

SAE_DIR=$WORK/sae                 # SAE checkpoints (B0 and ranking-aware runs)
B0=$SAE_DIR/B0/last.pt
LAT=$WORK/latents                 # exported concepts, one folder per SAE
RESULTS=$WORK/results             # small metric files, one per reported number
mkdir -p "$RAW" "$DATA" "$SAE_DIR" "$LAT" "$RESULTS"

log() { printf '\n== %s\n' "$*"; }
run() {  # print and run a command (or only print it under DRY_RUN=1)
    printf '+ %s\n' "$*"
    [ "${DRY_RUN:-0}" = 1 ] || "$@"
}
step() {  # output, command...: run the command unless the output already exists
    local out=$1; shift
    if [ -e "$out" ] && [ "${DRY_RUN:-0}" != 1 ]; then echo "skip (exists) $out"; else run "$@"; fi
}
need() { [ -e "$1" ] || [ "${DRY_RUN:-0}" = 1 ] || { echo "missing $1: run stage '$2' first" >&2; exit 1; }; }

# Export concepts for one SAE: passages (upstream JSONL) and every query split used later.
export_latents() {  # checkpoint, tag
    local ckpt=$1 out=$LAT/$2 split
    mkdir -p "$out"
    for split in train val dev dl19 dl20; do
        step "$out/$split.query.json" $PY -m p2.evaluate export --checkpoint "$ckpt" \
            --embeddings "$DATA/${split}_queries.npy" --output "$out/$split.query.json" --queries --device "$DEVICE"
    done
    [ -e "$out/validation.query.json" ] || run ln -sfn val.query.json "$out/validation.query.json"
    step "$out/passages.latent.jsonl" $PY -m p2.evaluate export --checkpoint "$ckpt" \
        --embeddings "$DATA/passages.npy" --output "$out/passages.latent.jsonl" --device "$DEVICE"
}

# Released CL-SR scoring (24 concepts per passage), then upstream evaluate.py, on dev, DL19 and DL20.
clsr_eval() {  # tag
    local out=$LAT/$1 split qrels
    mkdir -p "$out/fast_max24"
    for split in dev dl19 dl20; do
        qrels=$([ "$split" = dev ] && echo "$RAW/qrels.dev.small.tsv" || echo "$DATA/$split.qrels")
        step "$out/fast_max24/$split.tsv" $PY -m p2.clsr --passage-latents "$out/passages.latent.jsonl" \
            --query-latents "$out/$split.query.json" --output "$out/fast_max24/$split.tsv" --max-latents 24
        run ln -sf "$qrels" "$out/${split}_qrels.txt"  # upstream evaluate.py reads <split>_qrels.txt
        step "$RESULTS/clsr.$1.$split.txt" bash -c "cd '$out' && $PY '$REPO/upstream/Decoding-Dense-Embeddings/cl-sr/evaluate.py' \
            --eval_data $split --result_path '$out/fast_max24/$split.tsv' | tee '$RESULTS/clsr.$1.$split.txt'"
    done
}

# Table 2 reconstruction rows (NMSE, reconstructed dense MRR@10) for one SAE.
table1() {  # checkpoint, tag
    step "$RESULTS/table1.$2.json" bash -c "$PY -m p2.evaluate table1 --checkpoint '$1' --passages '$DATA/passages.npy' \
        --queries '$DATA/dev_queries.npy' --qrels '$RAW/qrels.dev.small.tsv' --device $DEVICE | tee '$RESULTS/table1.$2.json'"
}

# ---------------------------------------------------------------------------------------------------------
stage_setup() {
    log "setup"
    run git submodule update --init --recursive
    run $PY -m pip install -r requirements.txt pytrec_eval
}

stage_test() {
    log "unit tests"
    run $PY -m unittest discover -s tests -v
}

stage_smoke() {
    log "smoke tier (needs RAW with collection.tsv, queries.train.tsv, qrels.train.tsv)"
    run env RAW="$RAW" OUT="$WORK/smoke" PY="$PY" bash scripts/smoke.sh
}

stage_data() {
    log "data: MS MARCO, shared split (checked against splits/SHA256SUMS), TREC DL"
    run env OUT="$RAW" PY="$PY" bash scripts/split.sh
    local url=https://msmarco.z22.web.core.windows.net/msmarcoranking y
    # SimLM's titled corpus, which the released pipeline encodes (dense MRR@10 0.407 vs 0.393 untitled).
    step "$RAW/msmarco_bm25_official/passages.jsonl.gz" bash -c "cd '$RAW' && curl -fSL --retry 3 -O \
        https://huggingface.co/datasets/intfloat/simlm-msmarco/resolve/main/msmarco_bm25_official.zip && unzip -o msmarco_bm25_official.zip"
    for y in 2019 2020; do
        step "$RAW/msmarco-test$y-queries.tsv.gz" curl -fSL --retry 3 -o "$RAW/msmarco-test$y-queries.tsv.gz" "$url/msmarco-test$y-queries.tsv.gz"
        step "$RAW/${y}qrels-pass.txt" curl -fSL --retry 3 -o "$RAW/${y}qrels-pass.txt" "https://trec.nist.gov/data/deep/${y}qrels-pass.txt"
        step "$DATA/dl${y#20}.qrels" cp "$RAW/${y}qrels-pass.txt" "$DATA/dl${y#20}.qrels"
        step "$RAW/msmarco-test$y-queries.tsv" bash -c "gunzip -c '$RAW/msmarco-test$y-queries.tsv.gz' > '$RAW/msmarco-test$y-queries.tsv'"
    done

    log "data: SimLM embeddings and dense top-100 candidates with teacher scores"
    step "$DATA/passages.npy" $PY -m p2.encode passages --input "$RAW/msmarco_bm25_official/passages.jsonl.gz" \
        --output "$DATA/passages.npy" --batch-size 1024
    step "$DATA/train_queries.npy" $PY -m p2.encode queries --input "$RAW/split/train.queries.tsv" --output "$DATA/train_queries.npy"
    step "$DATA/val_queries.npy" $PY -m p2.encode queries --input "$RAW/split/validation.queries.tsv" --output "$DATA/val_queries.npy"
    step "$DATA/dev_queries.npy" $PY -m p2.encode queries --input "$RAW/queries.dev.small.tsv" --output "$DATA/dev_queries.npy"
    for y in 2019 2020; do
        step "$DATA/dl${y#20}_queries.npy" $PY -m p2.encode queries --input "$RAW/msmarco-test$y-queries.tsv" \
            --output "$DATA/dl${y#20}_queries.npy"
    done
    step "$DATA/val.cands.npy" $PY -m p2.candidates --passages "$DATA/passages.npy" --queries "$DATA/val_queries.npy" \
        --qrels "$RAW/split/validation.qrels" --output-prefix "$DATA/val"
    step "$DATA/train.cands.npy" $PY -m p2.candidates --passages "$DATA/passages.npy" --queries "$DATA/train_queries.npy" \
        --qrels "$RAW/split/train.qrels" --output-prefix "$DATA/train"
    local name
    for name in val_queries.npy val_queries.ids val.cands.npy val.teacher.npy val.pos.npy val.run; do
        [ -e "$DATA/${name/val/validation}" ] || run ln -sfn "$name" "$DATA/${name/val/validation}"  # sparsity tools read "validation_*"
    done
    # The scoring tools expect the split and dev qrels next to the embeddings.
    [ -e "$DATA/split" ] || run ln -sfn "$RAW/split" "$DATA/split"
    [ -e "$DATA/qrels.dev.small.tsv" ] || run ln -sfn "$RAW/qrels.dev.small.tsv" "$DATA/qrels.dev.small.tsv"
    step "$DATA/MANIFEST.sha256" bash -c "cd '$DATA' && sha256sum passages.npy ./*_queries.npy ./*_queries.ids ./*.cands.npy \
        ./*.teacher.npy ./*.pos.npy ./*.run ./*.qrels > MANIFEST.sha256"
}

stage_mismatch() {
    log "BM25 mismatch subsets (Pyserini, k1=0.9, b=0.4); compare with splits/p5_mismatch_k0.9_b0.4"
    run env PY="$BM25_PY" OUT="$RAW/bm25" DEV_TOPICS="$RAW/queries.dev.small.tsv" \
        VALIDATION_TOPICS="$RAW/split/validation.queries.tsv" bash scripts/bm25.sh
    local split qrels
    for split in dev validation; do
        qrels=$([ "$split" = dev ] && echo "$RAW/qrels.dev.small.tsv" || echo "$RAW/split/validation.qrels")
        step "$RAW/bm25/mismatch/$split.mismatch.json" $PY -m p5.mismatch --run "$RAW/bm25/bm25.$split.k0.9_b0.4.trec" \
            --qrels "$qrels" --output-dir "$RAW/bm25/mismatch" --name "$split"
    done
    run diff -q "$RAW/bm25/mismatch/dev.mismatch_1000.qids" splits/p5_mismatch_k0.9_b0.4/dev.mismatch_1000.qids
}

stage_b0() {
    need "$DATA/train.cands.npy" data
    log "B0: upstream SAE objective, per-latent decoder norm (the paper's), k=32, $B0_STEPS steps"
    step "$B0" $PY -m p2.train --data-dir "$DATA" --val-qrels "$RAW/split/validation.qrels" --output "$SAE_DIR/B0" \
        --rank-loss none --steps "$B0_STEPS" --k 32 --eval-every 20000 --in-ram --decoder-norm atom --device "$DEVICE"
    log "B0: Table 2 (NMSE, reconstructed dense MRR@10), latents, CL-SR on dev / DL19 / DL20"
    table1 "$B0" B0
    export_latents "$B0" B0
    clsr_eval B0
    log "B0: shared CL-SR index for query expansion (upstream format, 24 concepts per passage)"
    step "$LAT/B0/index.pkl" $PY upstream/Decoding-Dense-Embeddings/cl-sr/indexing.py \
        --passage_latents "$LAT/B0/passages.latent.jsonl" --index_save_path "$LAT/B0/index.pkl" --max_latents 24
    step "$LAT/B0/stats.json" $PY -m p1.prepare corpus --passages "$LAT/B0/passages.latent.jsonl" \
        --database "$LAT/B0/corpus.db" --stats "$LAT/B0/stats.json" --max-concepts 24
}

stage_scoring() {
    need "$LAT/B0/passages.latent.jsonl" b0
    local OUT=$WORK/scoring bundle=$WORK/scoring/bundle seed
    log "scoring: bundle layout the scoring tools read (links only)"
    run mkdir -p "$bundle"
    [ -e "$bundle/B0_final_atom.pt" ] || run ln -sfn "$B0" "$bundle/B0_final_atom.pt"
    [ -e "$bundle/data_titled" ] || run ln -sfn "$DATA" "$bundle/data_titled"
    [ -e "$bundle/latents_B0_final_atom" ] || run ln -sfn "$LAT/B0" "$bundle/latents_B0_final_atom"

    log "scoring: learned scorer, 5,000 train / 5,000 validation queries, 10 epochs, seeds 42-44"
    step "$OUT/s42/model.pt" $PY -m p1.pilot --bundle "$bundle" --output-dir "$OUT/s42" --train-limit 5000 \
        --validation-limit 5000 --max-concepts 24 --epochs 10 --seed 42 --device "$DEVICE"
    for seed in 43 44; do  # same prepared queries, different initialisation
        run mkdir -p "$OUT/s$seed"
        step "$OUT/s$seed/model.pt" $PY -m p1.train --train "$OUT/s42/train.jsonl" --validation "$OUT/s42/validation.jsonl" \
            --stats "$OUT/s42/stats.json" --qrels "$OUT/s42/validation.qrels" --output "$OUT/s$seed/model.pt" \
            --epochs 10 --seed "$seed" --device "$DEVICE"
    done
    [ "${DRY_RUN:-0}" = 1 ] || printf '{"42": "%s", "43": "%s", "44": "%s"}\n' \
        "$OUT/s42/model.pt" "$OUT/s43/model.pt" "$OUT/s44/model.pt" > "$OUT/models.json"

    log "scoring: full-corpus posting index, validation freeze (tuned-BM25 grid), held-out dev / DL19 / DL20"
    step "$OUT/index-cap24" $PY -m p1.index --passages "$LAT/B0/passages.latent.jsonl" --output-dir "$OUT/index-cap24" \
        --max-concepts 24 --batch-docs 50000
    step "$OUT/selection/freeze.json" $PY -m p1.final validation --index "$OUT/index-cap24" --query-dir "$LAT/B0" \
        --bundle "$bundle" --models "$OUT/models.json" --output-dir "$OUT/selection" --device "$DEVICE"
    step "$OUT/heldout/summary.json" $PY -m p1.final heldout --index "$OUT/index-cap24" --query-dir "$LAT/B0" \
        --bundle "$bundle" --freeze "$OUT/selection/freeze.json" --train-ids "$RAW/split/train.qids" \
        --validation-ids "$RAW/split/validation.qids" --output-dir "$OUT/heldout" --device "$DEVICE"
    step "$OUT/report" $PY -m p1.package --summary "$OUT/heldout/summary.json" --validation "$OUT/selection/freeze.json" \
        --output-dir "$OUT/report"
}

sae_train() {  # name, loss, space, weight, seed
    step "$SAE_DIR/${1}_s$5/best.pt" $PY -m p2.train --data-dir "$DATA" --val-qrels "$RAW/split/validation.qrels" \
        --init "$B0" --in-ram --output "$SAE_DIR/${1}_s$5" --rank-loss "$2" --rank-space "$3" --rank-weight "$4" \
        --steps 20000 --seed "$5" --lr 1e-5 --eval-every 2000 --score-norm zscore --device "$DEVICE"
}

stage_sae() {
    need "$B0" b0
    log "sae: validation sweep (20k steps, lr 1e-5; best.pt = best validation CL-SR with NMSE <= 1.1 x B0)"
    sae_train B1_recon_only none recon 0 42        # control: extra training only
    sae_train naacl_w0.1 naacl recon 0.1 42        # Kang et al. (2025) objective
    sae_train listwise_recon_w0.1 listwise recon 0.1 42
    sae_train listwise_latent_w0.1 listwise latent 0.1 42
    local w run seed
    for w in 0.05 0.1 0.2; do sae_train listwise_bm25_w$w listwise bm25 "$w" 42; done
    log "sae: selected run $SAE_SEL, seeds 43 and 44"
    for seed in 43 44; do sae_train "$SAE_SEL" listwise bm25 "${SAE_SEL##*_w}" "$seed"; done

    log "sae: dev / DL19 / DL20 for the controls and the three seeds, and concept stability against B0"
    for run in B1_recon_only_s42 naacl_w0.1_s42 "${SAE_SEL}_s42" "${SAE_SEL}_s43" "${SAE_SEL}_s44"; do
        table1 "$SAE_DIR/$run/best.pt" "$run"
        export_latents "$SAE_DIR/$run/best.pt" "$run"
        clsr_eval "$run"
    done
    for run in B1_recon_only_s42 naacl_w0.1_s42 "${SAE_SEL}_s42"; do
        step "$RESULTS/concepts.$run.json" bash -c "$PY -m p2.concepts --reference '$B0' --checkpoint '$SAE_DIR/$run/best.pt' \
            --passages '$DATA/passages.npy' --device $DEVICE | tee '$RESULTS/concepts.$run.json'"
    done
}

stage_sparsity() {
    need "$B0" b0
    local OUT=$WORK/sparsity tau min=8 max=128 k
    # tau = the SAE's own inference threshold (0.49723 for the report's B0), formatted as the stats folder name.
    tau=${SPARSITY_TAU:-$($PY -c "import sys, torch; print(f\"{torch.load(sys.argv[1], map_location='cpu')['meta']['threshold']:.5f}\")" "$B0" 2>/dev/null || echo 0.49723)}
    run mkdir -p "$OUT"
    log "sparsity: top-128 activation cache, fixed-k indexes k = 32, 48, 64, 128"
    step "$OUT/top128_cache" $PY -m p3.extract_topk_cache --checkpoint "$B0" --input-embeddings "$DATA/passages.npy" \
        --output-dir "$OUT/top128_cache" --k 128 --resume
    for k in 32 48 64; do
        step "$OUT/fixed$k.diskindex" $PY -m p3.build_fixed_disk --cache "$OUT/top128_cache" --output "$OUT/fixed$k.diskindex" --k "$k"
    done
    step "$OUT/fixed128.diskindex" $PY -m p3.build_fixed128_disk --cache "$OUT/top128_cache" --output "$OUT/fixed128.diskindex"

    log "sparsity: adaptive index (tau = SAE threshold $tau, Kmin = $min, Kmax = $max)"
    step "$OUT/adaptive_stats" $PY -m p3.compute_adaptive_corpus_stats --cache "$OUT/top128_cache" \
        --output "$OUT/adaptive_stats" --thresholds "$tau" --min-concepts "$min"
    step "$OUT/adaptive.diskindex" $PY -m p3.build_adaptive_disk --cache "$OUT/top128_cache" \
        --stats "$OUT/adaptive_stats/threshold_$tau" --output "$OUT/adaptive.diskindex" \
        --threshold "$tau" --min-concepts "$min" --max-concepts "$max"
    step "$OUT/adaptive.diskindex/.sorted" bash -c "$PY -m p3.sort_adaptive_shards --index-dir '$OUT/adaptive.diskindex' \
        && touch '$OUT/adaptive.diskindex/.sorted'"

    log "sparsity: full-index validation MRR@10 for every index (the report ran this method on validation only)"
    for k in fixed32 fixed48 fixed64 adaptive; do
        step "$RESULTS/sparsity.$k.validation.json" $PY -m p3.evaluate_fixed_disk --index "$OUT/$k.diskindex" \
            --queries "$LAT/B0/val.query.json" --qrels "$RAW/split/validation.qrels" \
            --checkpoint "$OUT/$k.progress.json" --output "$RESULTS/sparsity.$k.validation.json"
    done
    step "$RESULTS/sparsity.fixed128.validation.json" $PY -m p3.evaluate_fixed128_disk --index "$OUT/fixed128.diskindex" \
        --queries "$LAT/B0/val.query.json" --qrels "$RAW/split/validation.qrels" \
        --checkpoint "$OUT/fixed128.progress.json" --output "$RESULTS/sparsity.fixed128.validation.json"
}

stage_pruning() {
    need "$LAT/B0/passages.latent.jsonl" b0
    local OUT=$WORK/pruning lam seed kind split qrels
    run mkdir -p "$OUT/idx" "$OUT/gates" "$OUT/runs"
    log "pruning: full Lucene impact index (released k1 = 0.6, b = 2.0) and per-concept document frequencies"
    step "$OUT/idx/full" $BM25_PY -m p4.pyserini_indexing --latents "$LAT/B0/passages.latent.jsonl" --output "$OUT/idx/full"
    log "pruning: gate training data (each concept's Eq. 1 term for every train query x its 100 SimLM candidates)"
    step "$OUT/gate_X.npz" $PY -m p4.prepare_data --latents "$LAT/B0/passages.latent.jsonl" --queries "$LAT/B0/train.query.json" \
        --ids "$DATA/train_queries.ids" --cands "$DATA/train.cands.npy" --out "$OUT/gate_X.npz" --k1 0.6 --b 2.0 --k2 1.5
    log "pruning: gates for lambda in {0.1, 0.4, 0.8, 1.6} (seed 0), seeds 1-2 for the selected lambda 0.4"
    for lam in 0.1 0.4 0.8 1.6; do
        step "$OUT/gates/gate_lam$lam.drop.npy" $PY -m p4.gate_train --x "$OUT/gate_X.npz" --teacher "$DATA/train.teacher.npy" \
            --df "$OUT/idx/full/df.npy" --lam "$lam" --epochs 20 --lr 0.05 --seed 0 --device "$DEVICE" --out "$OUT/gates/gate_lam$lam"
    done
    for seed in 1 2; do
        run mkdir -p "$OUT/gates_s$seed"
        step "$OUT/gates_s$seed/gate_lam0.4.drop.npy" $PY -m p4.gate_train --x "$OUT/gate_X.npz" --teacher "$DATA/train.teacher.npy" \
            --df "$OUT/idx/full/df.npy" --lam 0.4 --epochs 20 --lr 0.05 --seed "$seed" --device "$DEVICE" --out "$OUT/gates_s$seed/gate_lam0.4"
    done
    log "pruning: DF baseline at the same share of removed entries, pruned indexes (original passage lengths kept)"
    run $PY -m p4.df_drops --df "$OUT/idx/full/df.npy" --gates "$OUT/gates/gate_lam*.drop.npy"
    local drops=()
    for lam in 0.1 0.4 0.8 1.6; do drops+=("gate_lam$lam=$OUT/gates/gate_lam$lam.drop.npy" "df_lam$lam=$OUT/gates/df_lam$lam.drop.npy"); done
    for seed in 1 2; do drops+=("gate_lam0.4_s$seed=$OUT/gates_s$seed/gate_lam0.4.drop.npy"); done
    for kind in "${drops[@]}"; do
        step "$OUT/idx/${kind%%=*}" $BM25_PY -m p4.pyserini_indexing --latents "$LAT/B0/passages.latent.jsonl" \
            --output "$OUT/idx/${kind%%=*}" --drop "${kind#*=}"
    done
    log "pruning: search dev / DL19 / DL20 on every index"
    for kind in full "${drops[@]%%=*}"; do
        for split in dev dl19 dl20; do
            qrels=$([ "$split" = dev ] && echo "$RAW/qrels.dev.small.tsv" || echo "$DATA/$split.qrels")
            step "$RESULTS/pruning.${split}_$kind.json" bash -c "$BM25_PY -m p4.pyserini_search --index '$OUT/idx/$kind' \
                --queries '$LAT/B0/$split.query.json' --qrels '$qrels' --output '$OUT/runs/${split}_$kind.trec' \
                > '$RESULTS/pruning.${split}_$kind.json'"
        done
    done
}

stage_expansion() {
    need "$LAT/B0/index.pkl" b0
    # scripts/p5.sh (the query-expansion runner) reads these, including its output folder as P5.
    export PY RAW DATA SAE=$B0 INDEX=$LAT/B0/index.pkl STATS=$LAT/B0/stats.json CORPUS_DB=$LAT/B0/corpus.db \
        P5=$WORK/expansion VAL_QRELS=$RAW/split/validation.qrels DEV_QRELS=$RAW/qrels.dev.small.tsv \
        MISMATCH=${MISMATCH:-$REPO/splits/p5_mismatch_k0.9_b0.4}
    local OUT=$WORK/expansion s b split seed
    row() { run env "$@" bash scripts/p5.sh; }
    log "expansion: expander training data and training (100k train queries, validation selection of B and checkpoint)"
    step "$OUT/train.groups.jsonl" env ROW=prepare TRAIN_SAMPLE=100000 bash scripts/p5.sh
    step "$OUT/expander.pt" env ROW=train bash scripts/p5.sh
    for seed in 1 2; do
        step "$OUT/expander_s$seed.pt" env ROW=train EXPANDER="$OUT/expander_s$seed.pt" TRAIN_ARGS="--seed $seed" bash scripts/p5.sh
    done
    log "expansion: validation rows B0, T (scale 0.5-0.9), N (B = 4, 8, 16), L; Holm family on validation"
    [ -e "$OUT/B0.val.json" ] || row ROW=B0
    for s in 0.5 0.6 0.7 0.8 0.9; do [ -e "$OUT/T$s.val.json" ] || row ROW=T PARAM=$s; done
    for b in 4 8 16; do [ -e "$OUT/N$b.val.json" ] || row ROW=N PARAM=$b; done
    [ -e "$OUT/L.val.json" ] || row ROW=L
    row ROW=family T_ROW="$T_ROW" N_ROW="$N_ROW"
    log "expansion: final one-shot runs with $T_ROW and $N_ROW on dev / DL19 / DL20, Holm family on dev, seeds 1-2"
    row ROW=prepare-final
    for split in dev dl19 dl20; do
        [ -e "$OUT/B0.$split.json" ] || row ROW=B0 SPLIT=$split FINAL=1
        [ -e "$OUT/$T_ROW.$split.json" ] || row ROW=T PARAM="${T_ROW#T}" SPLIT=$split FINAL=1
        [ -e "$OUT/$N_ROW.$split.json" ] || row ROW=N PARAM="${N_ROW#N}" SPLIT=$split FINAL=1
        [ -e "$OUT/L.$split.json" ] || row ROW=L SPLIT=$split FINAL=1
        for seed in 1 2; do
            [ -e "$OUT/L_s$seed.$split.json" ] || row ROW=L SPLIT=$split FINAL=1 NAME=L_s$seed EXPANDER="$OUT/expander_s$seed.pt"
        done
    done
    row ROW=family SPLIT=dev FINAL=1 T_ROW="$T_ROW" N_ROW="$N_ROW"
    log "expansion: diagnostics on validation (what L adds; found / lost relevant passages)"
    if [ -e "$RAW/collection.tsv" ]; then
        step "$RESULTS/expansion.diagnostics.json" $PY -m p5.example --queries "$OUT/val.query.json" \
            --expanded "$OUT/L.val.query.json" --embeddings "$DATA/val_queries.npy" --checkpoint "$B0" \
            --stats "$STATS" --expander "$OUT/expander.pt" --database "$CORPUS_DB" --baseline-run "$OUT/B0.val.run" \
            --run "$OUT/L.val.run" --qrels "$VAL_QRELS" --subset "$MISMATCH/validation.mismatch_1000.qids" \
            --query-text "$RAW/split/validation.queries.tsv" --describe "$OUT/val.query.json" \
            --describe-text "$RAW/split/validation.queries.tsv" --collection "$RAW/collection.tsv" \
            --output "$RESULTS/expansion.diagnostics.json"
    else
        echo "skip diagnostics: needs $RAW/collection.tsv (MS MARCO collection.tar.gz)"
    fi
}

# ---------------------------------------------------------------------------------------------------------
ALL=(setup test data b0 scoring sae sparsity pruning expansion)
[ $# -gt 0 ] || { sed -n '2,40p' "$0"; exit 2; }
for stage in "$@"; do
    case $stage in
    all) for s in "${ALL[@]}"; do "stage_$s"; done ;;
    setup|test|smoke|data|mismatch|b0|scoring|sae|sparsity|pruning|expansion) "stage_$stage" ;;
    *) echo "unknown stage '$stage' (see: bash scripts/reproduce.sh)" >&2; exit 2 ;;
    esac
done
log "done. Metric files: $RESULTS  (scoring: $WORK/scoring/heldout/summary.json, expansion: $WORK/expansion/*.json)"
