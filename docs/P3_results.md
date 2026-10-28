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

Still required: run `scripts/nscc/p3_encode_full.pbs` to export full-corpus
fixed k=32 and adaptive (`tau=3.5`, min=8, max=32) passages with the GPU, then
submit `scripts/nscc/p3_index_eval.pbs` separately for each variant on a
high-memory CPU node. Record corpus-wide metrics, serialized index sizes, and
query latencies. Use the same machine and query set for latency measurements.
Dev and TREC rows are final results and must not feed back into tuning.

| Variant | Split | MRR@10 | Recall@1000 | NDCG@10 | Mean concepts/passage | Index size (GiB) | Mean query latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| Fixed k=32 | Validation | — | — | — | — | — | — |
| Fixed k=48 | Validation | — | — | — | — | — | — |
| Fixed k=64 | Validation | — | — | — | — | — | — |
| Fixed k=128 | Validation | — | — | — | — | — | — |
| Adaptive (`tau=`, `K_min=`, `K_max=`) | Validation | — | — | — | — | — | — |
| Selected adaptive | MS MARCO dev | — | — | — | — | — | — |
| Selected adaptive | TREC DL 2019 | — | — | — | — | — | — |
| Selected adaptive | TREC DL 2020 | — | — | — | — | — | — |

## Limitations and interpretation

Validation's dense-candidate sweep is a threshold-screening proxy, not a full
retrieval result. Use the full CL-SR validation index for the selection rule.
Report negative or neutral results as measured. Include corpus subset size and
hardware if a full-corpus index or latency run was not feasible.
