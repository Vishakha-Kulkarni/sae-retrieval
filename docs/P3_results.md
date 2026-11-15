# P3: Adaptive sparsity per passage

## Method

We keep the released SAE checkpoint, query encoding, CL-SR BM25 scoring rule,
passage corpus, and inverted-index implementation fixed. For a passage embedding
`x`, the frozen encoder produces `a = W_enc (x - b_pre)`. P3 retains positive
activations above validation-tuned threshold `tau`, capped at `K_max`. If fewer
than `K_min` activations exceed `tau`, it fills from the strongest remaining
positive activations. Passages can therefore have different concept counts.
The checkpoint and decoder are not trained or changed.

We choose the setting with the smallest mean passage concept count whose full
validation-index MRR@10 is no more than 0.005 below the best fixed-k validation
result. Thresholds are selected only on the 5,000-query training-derived
validation split. MS MARCO dev and TREC DL 2019/2020 are reserved for final
evaluation.

## Base method

The released pipeline encodes dense SimLM passage and query vectors with an
overcomplete sparse autoencoder. Its linear encoder maps each embedding to
latent activations, BatchTopK training selects positive activations across the
training batch, and a tied linear decoder reconstructs the input vector. At
inference, the released encoder applies the checkpoint's activation threshold
and emits active concept IDs with their activation weights. CL-SR indexes
passage concepts and weights in inverted posting lists, then applies its BM25
scoring rule to query and passage concepts that overlap.

Before submission, record the exact SAE checkpoint, BatchTopK `k`, expansion
factor, stored inference threshold, SimLM checkpoint, and CL-SR constants used
in the team's baseline. Do not infer these values from the paper when the
released artifact or run configuration is available.

## Results

### Validation screening result

The earlier screening outputs are superseded because they changed query
sparsity along with passage sparsity. The corrected sweep held query activations
fixed at the checkpoint's stored threshold (3.78881) and compared against fixed
k=32. It used the 5,000 validation queries, each with 100 dense candidates, and
a 200,000-passage sample for CL-SR corpus statistics. This remains a candidate
reranking proxy, not full-index evaluation; its `recall_at_1000` is limited by
the 100 supplied candidates.

| Variant | MRR@10 | NDCG@10 | Mean concepts/passage |
|---|---:|---:|---:|
| Fixed k=32 | 0.32160 | 0.38418 | 32.00 |
| Adaptive, `tau=3.5`, min=8, max=32 | 0.31866 | 0.38023 | 30.87 |

The adaptive setting is within 0.005 MRR@10 of fixed k=32 and reduces estimated
mean concepts by about 3.5%. This is a small proxy gain; verify it with the full
CL-SR index before treating it as a result.

### Full-index results

The full-corpus CL-SR indexes were built and evaluated on the 5,000-query
validation split. These are validation measurements, not final dev or TREC
results. The adaptive configuration shown is `tau=0.49723`, `K_min=8`,
`K_max=128`; the threshold and limits were selected using validation data.
Latency is the mean over the 5,000 timed validation queries for each run.

| Variant | Split | MRR@10 | Recall@1000 | NDCG@10 | Mean concepts/passage | Index size (GiB) | Mean query latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| Fixed k=32 | Validation | 0.34268 | 0.96225 | 0.40289 | 32.00 | 2.64 | 988.45 |
| Fixed k=48 | Validation | 0.34722 | 0.96632 | 0.40940 | 48.00 | 3.96 | 1,205.55 |
| Fixed k=64 | Validation | 0.35230 | 0.96595 | 0.41528 | 64.00 | 5.28 | 1,262.21 |
| Fixed k=128 | Validation | 0.35575 | 0.96720 | 0.41894 | 127.87 | 10.54 | 1,530.76 |
| Adaptive (`tau=0.49723`, `K_min=8`, `K_max=128`) | Validation | 0.33352 | 0.96073 | 0.39392 | 32.58 | 2.69 | 995.06 |

The selected adaptive setting does **not** meet the planned quality target. Its
MRR@10 is 0.00915 below fixed k=32 and 0.02223 below the best fixed-k setting
(k=128), exceeding the allowed 0.005 gap. It also produces a slightly larger
index than fixed k=32 (2.69 vs. 2.64 GiB), with similar measured latency. Report
this as a negative validation result; it does not support a quality-preserving
index reduction. These results cover all 8,841,823 passages and 5,000 validation
queries. No MS MARCO dev or TREC DL measurements are recorded here.

The validation outputs are stored under
`sae-simlm-k32-baseline/data_titled/p3_full_eval/`: the fixed-k disk runs are
`fixed32.disk.validation.json`, `fixed48.disk.validation.json`, and
`fixed64.disk.validation.json`; fixed k=128 is `fixed128.validation.json`; and
the adaptive run is `adaptive_t0.49723.validation.v2.json`.

## Limitations and interpretation

The earlier dense-candidate sweep is a threshold-screening proxy, not a full
retrieval result; the table above is the full-corpus validation evaluation.
Report negative or neutral results as measured. Final MS MARCO dev and TREC DL
evaluation remain team-level final-evaluation tasks and are not reported here.
