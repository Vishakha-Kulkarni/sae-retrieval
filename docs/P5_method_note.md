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
on the 988 dev queries where BM25 fails at top 1000, CL-SR Max reaches MRR@10 0.143 against SimLM's 0.185
(paper Table 5).

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

**What counts as an improvement (P5 target):** L beats B0 on validation Recall@1000 and on validation
mismatch@1000 MRR@10, significant under a paired randomisation test (sign flips, 10,000 permutations,
p < 0.05). Each row is tested once against B0, so no multiple-comparison correction is applied. L against T
and N is reported side by side; `python -m p5.harness compare` gives their p-values, which are exploratory.
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
  top-1000 count is outside 900–1,100 (paper: 988). Setting used and our count: *to record after the run*.

## P3 boundary
Near-miss concepts make part of the expander a learned per-query threshold, which is why T exists. P3's
adaptive threshold acts on passages only. If P3 also changes queries, the combined run applies P3's
threshold first and P5's expansion on top.

## Run order
Paths come from `scripts/nscc/env.sh` (`SAE`, `INDEX`, `STATS`, `CORPUS_DB`, `P5`, `MISMATCH`). `STATS` and
`CORPUS_DB` come from `p1.prepare corpus --max-concepts` equal to the index's `--max_latents`.
1. Mismatch subsets (once, local, Java 21): `VALIDATION_TOPICS=$RAW/split/validation.queries.tsv OUT=$RAW/bm25
   bash scripts/bm25.sh`, then `python -m p5.mismatch --run … --name validation` and `--name dev`.
2. `qsub -v ROW=prepare scripts/nscc/p5.pbs` (query latents, training groups), then `ROW=train`.
3. `ROW=B0`, then `ROW=T,PARAM=0.9` … `0.5`, `ROW=N,PARAM=4|8|16`, `ROW=L` (all on validation).
4. Pick the best T scale and N budget on validation; L's budget is in its checkpoint.
5. After the freeze: `ROW=prepare-final`, then each chosen row with `SPLIT=dev|dl19|dl20,FINAL=1`; 3 training
   seeds for L (`TRAIN_ARGS="--seed …"`, `EXPANDER=…`).
6. `python -m p5.table` over every piece's row JSONs builds the ablation table.

The smoke tier (`RAW=… scripts/smoke.sh`) runs the same `scripts/p5.sh` steps on 100k passages.
