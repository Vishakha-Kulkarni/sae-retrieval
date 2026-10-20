# P2: Ranking-aware SAE training (method note)

## What we start from (unchanged upstream code)
- **SAE:** the released BatchTopK checkpoint (`sae_32_k32`, 768 → 24,576 latents, k = 32), from
  `upstream/Decoding-Dense-Embeddings/sae`. Its objective is reconstruction MSE plus a dead-latent auxiliary term.
- **Retrieval:** upstream CL-SR (`cl-sr/indexing.py`, `cl-sr/latent_retrieve.py`, BM25 with k1 = 0.6, b = 2.0), used as is.
- **Our one change:** an extra loss term during SAE training. Everything downstream (latent extraction
  format, threshold encoding, index, BM25 scoring, evaluation) stays the released code.

Why there is a `p2/sae.py`: upstream `sae/train.py` does not run as shipped (it uses undefined `device`,
`lm`, `k` and `batch_size`, and calls `wandb.finish()` with wandb disabled), and `sae/models.py` hard-codes CUDA.
`p2/sae.py` and `p2/train.py` fix those bugs only. They keep the upstream parameter names (`linear`, `linear2`,
`bias_pre`), the BatchTopK step, the auxiliary loss (α = 1/32, dead after 20 steps), the decoder
normalisation and the threshold encoding. Released checkpoints load into our trainer, and our
checkpoints load into upstream `extract_latent_concepts.py`.

## Problem
The SAE is trained to rebuild each vector. Nothing asks its concepts to keep relevant passages above
irrelevant ones, and CL-SR ranks purely on those concepts.

## Method
loss = MSE + aux + λ · var(x) · KL( softmax(s_teacher / T) ‖ softmax(s_student / T) )

- **Teacher:** SimLM dot products between a training query and its dense top-100 candidates
  (8 fixed top ranks plus 24 sampled from the rest at each step).
- **Student, `recon` space:** the same dot products, computed on the SAE reconstructions.
- **Student, `latent` space (main contribution):** dot products of the sparse concept codes, which is the
  representation CL-SR actually searches, with a learned scalar scale.
- var(x) keeps λ on the same scale as the normalised reconstruction error. λ ∈ {0.1, 0.5, 1.0}.

## How this differs from NAACL 2025 (Kang et al.)
Their Eq. 2 applies KL on reconstructions over each query's **relevant passages**, normalised within the
batch. We distil the dense model's **full ranking over hard candidates**, and we apply the loss in
**concept space**. Their objective is implemented as the `naacl` baseline under the same budget.

## Comparison: what each row is compared against
All rows use the same embeddings, the same released CL-SR index code, the same splits and the same threshold
procedure. Tuning and checkpoint selection use the 5,000-query validation split only. MS MARCO dev and TREC DL
are used once, for the final numbers.

| Row | What it is | Purpose |
|---|---|---|
| B0 | Released SAE, no training | **The baseline we must beat** |
| B1 | B0 + the same fine-tuning steps, reconstruction only | Control: shows that any gain is not just extra training |
| B2 | B0 + NAACL KL (λ sweep) | Strongest prior method for this idea |
| Ours-R | B0 + listwise KL, recon space (λ sweep) | Effect of hard-candidate listwise distillation |
| Ours-L | B0 + listwise KL, latent space (λ sweep) | Effect of applying the loss where CL-SR searches |

**What counts as an improvement (P2 target):** CL-SR MRR@10 on MS MARCO dev is higher than **B0 and B1**,
significant under a paired randomisation test (p < 0.05), with NMSE ≤ 1.10 × NMSE(B0).
Also reported: Recall@1000, NDCG@10 on TREC DL 19/20, active concepts per passage, index size,
dead-latent share, and a spot check that concept descriptions stay coherent.

## Run order
1. `scripts/nscc/encode.pbs` (shared week-1 task: embeddings and top-100 candidates; also used by P1).
2. `python -m p2.evaluate metrics --checkpoint $RELEASED --threshold ...` on validation gives the B0 numbers.
3. `PROJECT_ID=... bash scripts/nscc/submit_sweep.sh` (10 jobs).
4. Pick the best λ per family on validation (`best.pt` already applies the NMSE constraint).
5. `python -m p2.evaluate export ...` for passages and dev queries, then upstream `cl-sr/indexing.py`,
   `latent_retrieve.py` and `evaluate.py` give the final dev and TREC numbers. Run 3 seeds for the winner.
