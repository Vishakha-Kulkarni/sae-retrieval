# P3: Adaptive sparsity per passage

P3 changes passage encoding only. It uses the released SAE checkpoint and its
encoder activations, then retains activations above a scalar threshold with a
minimum and maximum concept count per passage. The resulting sparse features
keep the upstream CL-SR JSONL schema (`docid`, `ids`, `weight`) and can be fed
to the existing inverted-index builder.

`adaptive_encoder.py` provides `encode_adaptive_batch(model, embeddings, ...)`.
It accepts the released-compatible SAE model from `p2.sae` and returns rows
of `[batch_index, concept_ids, activation_weights]`, plus an optional
reconstruction. The extraction runner reads the float32 `.npy` embeddings
produced by `p2.encode` lazily, so it does not load the whole corpus into RAM.

The extraction runner can be used after initializing the upstream submodule:

```bash
python -m p3.extract_passage_latents \
  --checkpoint PATH_TO_SAE_CHECKPOINT \
  --input-embeddings PATH_TO_PASSAGES_NPY \
  --output results/p3_adaptive_passages.jsonl \
  --mode adaptive --threshold VALIDATION_TUNED_THRESHOLD \
  --min-concepts 8 --max-concepts 128
```

Use `--max-docs 1000000` for a development run on a prefix of the corpus. For
fixed-k comparisons, run the same script with `--mode fixed --k 32` (and repeat
for 48, 64, and 128). Each output file can be indexed with the upstream
`cl-sr/indexing.py`; for adaptive output, leave its `--max_latents` unset so it
does not silently truncate selections a second time.

## Validation screening

Once the team's training-derived validation files are prepared, screen
thresholds against fixed-k baselines with:

```bash
python -m p3.validation_sweep \
  --checkpoint PATH_TO_SAE_CHECKPOINT \
  --data-dir PATH_TO_VALIDATION_DATA \
  --qrels PATH_TO_VALIDATION_QRELS \
  --thresholds 0.05,0.1,0.15,0.2 \
  --min-counts 8,16 --max-counts 64,128 \
  --output results/p3_validation_sweep.json
```

The data directory follows P2's validation layout: `passages.npy`,
`validation_queries.npy`, `validation.cands.npy`, and `validation_queries.ids`.
The sweep evaluates CL-SR BM25 on fixed dense candidate lists and estimates
corpus document frequencies and mean concept counts from a deterministic
passage sample. It is a screening proxy; use the full-index evaluation below to
choose/report final validation numbers. Never pass dev or TREC qrels to the
sweep when tuning.

## Full-index evaluation

Extract passage latents for adaptive and fixed k = 32, 48, 64, 128, then query
latents once using the unchanged released query threshold. Build and score each
variant against the training-derived validation qrels:

```bash
python -m p3.evaluate_index \
  --passage-latents results/p3_adaptive_passages.jsonl \
  --query-latents data/validation.query.json \
  --qrels data/validation.qrels \
  --output results/p3_adaptive_validation.json
```

This calls the pinned upstream CL-SR index builder and BM25 scorer. Its report
records MRR@10, Recall@1000, NDCG@10, mean concepts per passage, serialized
pickle index size, index build time, and mean/median query latency. Run the
same command for each fixed-k file. For adaptive output, do not set
`--max-latents`: that would truncate the selected concepts again. Keep machine,
query encoding, scoring constants, corpus, and index code fixed when comparing
latency and size. Time the same validation query set for each variant.

For a cluster node with sufficient RAM, `scripts/nscc/p3_full_eval.pbs` runs
the fixed k=32 and screened adaptive (`tau=3.5`, min=8, max=32) full-corpus
validation comparison sequentially. Copy the checkpoint and required files to
the cluster paths, then submit that PBS job. The upstream Python-object index
needs a high-memory node; this workstation has 16 GiB RAM and is not suitable
for building the full index safely.

## Selection rule

For passage embedding `x`, the frozen SAE encoder produces activations
`a = W_enc (x - b_pre)`. Keep each activation strictly greater than threshold
`tau`, capped at `max_concepts`. If fewer than `min_concepts` pass, fill from
the strongest remaining positive activations. If a passage has fewer than the
minimum positive activations in total, it retains those available. Unlike fixed
top-k, the number retained can vary by passage. Tune `tau` and the bounds using only the
5,000-query training-derived validation split; use validation retrieval quality
to choose the smallest mean concept count whose MRR@10 is within 0.005 of the
best fixed-k validation result. Then report MS MARCO dev and TREC DL only as
final evaluation.

The comparison should include fixed k = 32, 48, 64, and 128, as required by the
project plan. For each variant record MRR@10, Recall@1000, mean concepts per
passage, serialized index size, and query latency. Keep the SAE checkpoint,
query encoding, scoring function, corpus, and index implementation fixed.

## Base method notes for the report

The report-ready P3 method, base-method description, and result table template
are in [`docs/P3_results.md`](../docs/P3_results.md).

The released pipeline encodes dense SimLM passage and query embeddings with a
sparse autoencoder. The SAE's learned linear encoder maps the input embedding
to an overcomplete latent vector; BatchTopK training selects a fixed number of
positive latent activations across each training batch, and a tied linear
decoder reconstructs the dense embedding. At inference, `encode_batchtopk`
uses the checkpoint's stored `mean_of_min_act` threshold (or the explicitly
provided equivalent) and emits active latent IDs with activation weights.
CL-SR stores passage latent IDs and weights in an inverted index, then scores
passages with BM25 over shared latent concepts. Keep the query encoding,
checkpoint, BM25 constants, corpus, and index builder unchanged for P3.

This implementation extends only passage-time latent selection; it does not
retrain the SAE or alter query encoding or CL-SR scoring. Confirm the exact
training and checkpoint settings against the team's frozen checkpoint before
finalizing this description.

## Upstream source

The official upstream source is included as a Git submodule at
`upstream/Decoding-Dense-Embeddings` and is Apache-2.0 licensed. Initialize it
with `git submodule update --init --recursive` after cloning this repository.
