# P5: Query concept expansion (method note)

## What we start from (unchanged upstream code)
- **SAE, index and scoring:** the team's frozen SAE (`$SAE`, B0 from `scripts/nscc/baseline.pbs`), its passage
  latents and the CL-SR index from upstream `cl-sr/indexing.py`, scored by `cl-sr/latent_retrieve.py`
  (`retrieve_one`, BM25 with k1 = 0.6, k2 = 1.5, b = 2.0, BM25 IDF, length = indexed concept count).
- **Our one change:** extra concepts appended to each **query's** latents. Passages, the index, the BM25
  constants and the SAE stay as released. Original query concepts and weights are never changed (P1 owns
  reweighting), and every expanded query file has upstream's query JSON format.

Upstream bug found while building the row evaluator: `indexing.py` keys postings by `str(concept)`, but
`latent_retrieve.py main()` passes the JSON's integer IDs, so every lookup misses and every query returns
nothing. `p5.evaluate` calls the same `retrieve_one` with string IDs, as `tests/test_upstream.py` does.
Anyone running upstream `main()` directly gets empty runs.

## Problem
CL-SR scores a passage only through concepts it shares with the query. Short queries activate few concepts,
so a relevant passage expressed through related concepts can score zero. Vocabulary-mismatch queries show it:
on the 988 dev queries where BM25 fails at top 1000, CL-SR Efficient (our B0 configuration) reaches MRR@10
0.124 against SimLM's 0.185 (paper Table 5).

## Method
- **Pool (per query):** the 32 strongest *near-miss* concepts (positive SAE pre-activation at or below the
  threshold) plus the 8 nearest decoder neighbours of each active concept (cosine of `linear2.weight`
  columns, normalised explicitly since `p2/sae.py` normalises rows). Dropped: concepts already in the query,
  concepts missing from the index, and concepts with IDF ≤ 0, which would only lower scores.
- **Expander:** a 2-layer MLP (hidden 32, about 1.3k parameters) maps each pool concept's features
  (pre-activation, cosine to the SimLM query vector, weighted max cosine to the active concepts, IDF, log
  active count) to a softplus weight.
- **Training (`p5.train`):** the student score is the released CL-SR BM25 of the expanded query over the
  query's SimLM dense top-100 candidates (`p5/bm25.py`, tested against upstream `retrieve_one`), times a
  learned scale. Loss = T² · KL(softmax(s_SimLM/T) ‖ softmax(s_student/T)) + λ_L1 · Σ added weight. Uses
  100k train queries; T = 1, λ_L1 = 1e-3, Adam lr 1e-3, batch 32, 2,000 steps. The training passages carry
  the index's concept cap, so the student scores what retrieval scores.
- **Inference:** keep the top-B pool concepts. B ∈ {4, 8, 16} and the checkpoint are chosen together by
  top-100 re-ranking MRR@10 on validation. The budget bounds the extra postings each query touches.

## Comparison: what each row is compared against
All rows use the same SAE, index, splits and evaluator (`p5.evaluate`: upstream `retrieve_one` over the full
index, top 1000). Tuning uses the 5,000-query validation split only. Dev and TREC DL run once, after the freeze.

| Row | Query side | Compared against | Purpose |
|---|---|---|---|
| B0 | Released threshold encoding | — | **The baseline we must beat** |
| T | Threshold × s, s ∈ {0.9, …, 0.5} tuned on validation | B0 | Control: is the gain just a lower threshold? |
| N | B0 + decoder neighbours, weight = source weight × cosine, B tuned | B0 | Rule-based expansion |
| L | B0 + learned expansion, B tuned | B0; reported next to T and N | P5 method |

**Choosing T and N (fixed 9 Nov, before their results):** the strongest control on P5's claim metric, i.e. the
scale s and budget B with the highest validation mismatch@1000 MRR@10, ties broken by validation Recall@1000.

**B0 on validation (9 Nov):** MRR@10 0.3336, Recall@1000 0.9605, NDCG@10 0.3936; mismatch@10/100/1000 MRR@10
0.214 / 0.148 / 0.111 (3,196 / 1,723 / 684 queries); 23.2 concepts per query. P4's Lucene B0 on the same split:
0.3347 / 0.9629 / 0.3952.

**Validation rows (10 Nov)** (MRR@10 / R@1k / mismatch@1000 MRR@10 / concepts per query; p vs B0 uncorrected):
L 0.3405 / 0.9607 / 0.1193 (p 0.010) / 39.2; T0.5 0.3397 / 0.9610 / 0.1171 (p 0.22) / 57.0 (best T is the grid
edge); N4 0.2851 / 0.9431 / 0.0896 / 27.2 (every N row is below B0). Chosen controls: **T0.5, N4**.
Validation family: L > B0 on mismatch (p_Holm 0.040), not on R@1k (1.0); L vs T0.5 not significant; L > N4.

**Dev (10 Nov, final):** B0 0.3158 / 0.9472 / 0.1038 / 23.9 / 385 ms; L 0.3201 / 0.9488 / 0.1068 / 39.9 / 442 ms;
T0.5 0.3215 / 0.9522 / 0.1099 / 58.7 / 446 ms; N4 0.2741 / 0.9285 / 0.0893 / 27.9 / 402 ms.
Dev family (Holm): L vs B0 R@1k 0.24, mismatch 0.40 (**not significant: the P5 target is not met**); L vs T0.5
R@1k 0.004 **in T's favour**, mismatch 0.40; L > N4 on both (≤ 0.006).
Postings per query vs B0: L 1.11×, T0.5 1.11×, N4 1.00×. TREC DL NDCG@10 (B0 / L / T0.5 / N4): DL19 0.611 /
0.611 / 0.611 / 0.522; DL20 0.568 / 0.574 / 0.569 / 0.512; only N differs from B0 (p < 0.01).
Seeds s1, s2 (dev, DL19, DL20): jobs 26194803–08. L over 3 seeds (mean ± sd): dev MRR@10 0.3198 ± 0.0003,
R@1k 0.9491 ± 0.0003, mismatch 0.1064 ± 0.0010, NDCG@10 0.3787 ± 0.0004; DL19 NDCG@10 0.6125 ± 0.0035, DL20
0.5723 ± 0.0028. T0.5 beats every seed on dev R@1k and mismatch. All three
checkpoints chose B = 16 (best proxy step 1000 / 1400 / 1600 of 2000).

**Diagnostics (`python -m p5.example`, validation, seed-42 L):** added concepts are 81% near-miss only, 13% both,
6% neighbour only (by weight 82 / 12 / 6%). First relevant rank, all 5,000: up 1,386, down 1,168, same 2,251,
found 11, lost 11, missed by both 173; mismatch@1000 subset: up 268, down 174, same 125, found 5, lost 4,
missed 108. All 184 B0 misses share an indexed concept with the original query; 106 contain an added concept.
Example: 515294 "the definition of protection", rank 5 → 1; top added concepts are question-form near misses,
one of five in the relevant passage.

**What counts as an improvement (P5 target):** L beats B0 on Recall@1000 and on mismatch@1000 MRR@10,
significant under a paired randomisation test (sign flips, 10,000 permutations) after Holm's correction.
**Formal family (`ROW=family`, `python -m p5.harness family`):** L vs B0, L vs T and L vs N, each on those
two metrics: 6 tests, Holm-corrected together at α = 0.05. "L beats the threshold control" and "L beats the
neighbour rule" are claimed only if their corrected p-values pass. Every other p-value (the † cells of the
ablation table, other rows vs B0, NDCG on DL, mismatch@10/100) is uncorrected and labelled as such.
If L does not beat T or N, it stays in the table with the numbers and an explanation. Also reported: NDCG@10
on TREC DL 19/20, mismatch@10/100, average query concepts, postings touched, latency (mean, p50, p95).

## Evaluation details that differ from upstream
- **Denominator:** every qrels query counts and a query missing from the run scores zero. Upstream
  `evaluate.py` averages only over queries in the run, and CL-SR writes nothing for a query with no indexed
  concept. Expansion removes such empty queries, so the upstream denominator would favour P5.
- **NDCG gain:** linear (trec_eval / `pytrec_eval`). It differs from P1's exponential default only on graded
  TREC DL qrels.
- **Mismatch subsets:** Pyserini BM25 top 1000 over the prebuilt `msmarco-v1-passage` index
  (`scripts/bm25.sh`), default k1 = 0.9, b = 0.4; the tuned k1 = 0.82, b = 0.68 is the fallback if the dev
  top-1000 count is outside 900–1,100 (paper: 988). **Used: default k1 = 0.9, b = 0.4** (Pyserini 2.4.0,
  `scripts/nscc/bm25.pbs`, 3 Nov). Dev mismatch@10/100/1000 = 4,289 / 2,327 / 988 of 6,980 (paper 4.3k / 2.3k
  / 988); validation = 3,196 / 1,723 / 684 of 5,000. The tuned setting gave 4,198 / 2,243 / 957 on dev.

## P3 boundary
Near-miss concepts make part of the expander a learned per-query threshold, which is why T exists. P3's
adaptive threshold acts on passages only. If P3 also changes queries, the combined run applies P3's
threshold first and P5's expansion on top.

## Run order
Paths come from `scripts/nscc/env.sh` (`SAE`, `INDEX`, `STATS`, `CORPUS_DB`, `P5`, `MISMATCH`). `STATS` and
`CORPUS_DB` come from `p1.prepare corpus --max-concepts` equal to the index's `--max_latents`.
1. Mismatch subsets (once): `bash scripts/nscc/bm25_setup.sh` on a login node, then `qsub scripts/nscc/bm25.pbs`
   (BM25 runs and `p5.mismatch` for both settings), then the `ln -sfn mismatch.<tag> $RAW/bm25/mismatch` it prints.
2. `qsub -v ROW=prepare scripts/nscc/p5.pbs` (query latents, training groups), then `ROW=train`.
3. `ROW=B0`, then `ROW=T,PARAM=0.9` … `0.5`, `ROW=N,PARAM=4|8|16`, `ROW=L` (all on validation).
4. Pick the best T scale and N budget on validation; L's budget is in its checkpoint. Then
   `ROW=family,T_ROW=T0.7,N_ROW=N8` (with the chosen names) gives the validation result.
5. After the freeze: `ROW=prepare-final`, then each chosen row with `SPLIT=dev|dl19|dl20,FINAL=1`, and
   `ROW=family,SPLIT=dev,FINAL=1`; 3 training seeds for L (`TRAIN_ARGS="--seed …"`, `EXPANDER=…`).
6. `python -m p5.table` over every piece's row JSONs builds the ablation table.

The smoke tier (`RAW=… scripts/smoke.sh`) runs the same `scripts/p5.sh` steps on 100k passages.
