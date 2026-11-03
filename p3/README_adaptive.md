# P3 adaptive sparsity: experiment record

This note accompanies the reusable extraction and disk-index workflow in
[`README.md`](README.md). It records the validation configuration and points to
the result files committed with this experiment.

## Configuration and evaluation scope

- Corpus: 8,841,823 passages.
- Evaluation: the 5,000-query training-derived validation split.
- SAE/query encoding: frozen released checkpoint; query threshold
  `0.49722998729348183`.
- Adaptive passage rule: activation threshold `0.49723`, minimum 8 concepts,
  maximum 128 concepts.
- Fixed-k comparison: `k = 32, 48, 64, 128`.
- Selection tolerance: adaptive MRR@10 within 0.005 of the best fixed-k
  validation result.

The sweep JSON is a candidate-reranking proxy with full-corpus BM25 statistics;
its recall is limited by the 100 supplied candidates. The `.validation.json`
files are full-corpus index evaluations over the complete validation query set.
Neither is MS MARCO dev or TREC DL evaluation.

## Result artifacts

All files are under
`sae-simlm-k32-baseline/data_titled/p3_full_eval/`:

- `adaptive_validation_sweep_exact.json`: threshold-screening proxy and
  selection record.
- `adaptive_exact_stats/threshold_0.49723/metadata.json` and
  `adaptive_t0.49723.diskindex/metadata.json`: selected adaptive statistics and
  index configuration (index paths in result files are relative to this
  directory so they remain portable).
- `adaptive_t0.49723.validation.v2.json`: full-index adaptive validation
  result.
- `fixed32.disk.validation.json`, `fixed48.disk.validation.json`, and
  `fixed64.disk.validation.json`: full-index fixed-k disk results.
- `fixed128.validation.json`: full-index fixed k=128 result.

The adaptive run scored MRR@10 0.33352 at 2.69 GiB; fixed k=32 scored 0.34268
at 2.64 GiB, and the best fixed-k result (k=128) scored 0.35575. Adaptive
sparsity therefore missed both the 0.005 quality tolerance and the target of a
smaller index than fixed k=32. Report this as a negative result. The combined
table and interpretation are in [`../docs/P3_results.md`](../docs/P3_results.md).

Rebuild instructions are in [`README.md`](README.md). Index shards, activation
caches, query runs, and checkpoints are generated artifacts and are not part of
this result-file commit.
