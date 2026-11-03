# P3: Adaptive sparsity per passage

P3 changes passage encoding only. It uses the released SAE checkpoint and
encoder activations, retaining activations above a validation-tuned threshold
with a minimum and maximum concept count per passage. Query encoding, the SAE
checkpoint, and CL-SR BM25 scoring stay fixed.

`adaptive_encoder.py` and `extract_passage_latents.py` produce sparse rows in the
upstream CL-SR JSONL schema (`docid`, `ids`, `weight`). The extraction runner
reads float32 `.npy` embeddings lazily. `--resume` validates existing output
records and continues at the next passage after an interrupted run.

## Validation procedure

Use only the 5,000-query training-derived validation split for threshold
selection. Never tune with MS MARCO dev or TREC DL qrels. The quick candidate
sweep in `validation_sweep.py` / `validation_sweep_exact.py` is a screening
proxy; final P3 comparisons use the full-corpus disk index and full validation
query set.

The full-corpus workflow uses a top-128 passage activation cache. It can be
created with:

```bash
python -m p3.extract_topk_cache \
  --checkpoint PATH_TO_SAE_CHECKPOINT \
  --input-embeddings PATH_TO_PASSAGES_NPY \
  --output-dir PATH_TO_P3_OUTPUT/top128_cache \
  --k 128 --resume
```

Build fixed-k indexes from that cache for `k=32, 48, 64, 128`:

```bash
python -m p3.build_fixed_disk \
  --cache PATH_TO_P3_OUTPUT/top128_cache \
  --output PATH_TO_P3_OUTPUT/fixed32.diskindex --k 32
```

Repeat with the required `k` and output name for each baseline. The dedicated
`build_fixed128_disk.py` is also available for k=128. Both builders create a
sharded on-disk index without loading the corpus index into Python memory.

For an adaptive setting, first compute full-corpus document frequencies and
average document length. These must use the same threshold and bounds as the
index build:

```bash
python -m p3.compute_adaptive_corpus_stats \
  --cache PATH_TO_P3_OUTPUT/top128_cache \
  --output PATH_TO_P3_OUTPUT/adaptive_exact_stats \
  --thresholds 0.49723 --min-concepts 8

python -m p3.build_adaptive_disk \
  --cache PATH_TO_P3_OUTPUT/top128_cache \
  --stats PATH_TO_P3_OUTPUT/adaptive_exact_stats/threshold_0.49723 \
  --output PATH_TO_P3_OUTPUT/adaptive_t0.49723.diskindex \
  --threshold 0.49723 --min-concepts 8 --max-concepts 128

python -m p3.sort_adaptive_shards \
  --index-dir PATH_TO_P3_OUTPUT/adaptive_t0.49723.diskindex
```

Sorting is required before retrieval: each posting shard must be globally
ordered by concept ID and document ID for binary-search lookup.

Export validation query latents once using the frozen SAE checkpoint threshold,
then evaluate each index with the same queries and validation qrels:

```bash
python -m p2.evaluate export \
  --checkpoint PATH_TO_SAE_CHECKPOINT \
  --embeddings PATH_TO_VALIDATION_QUERIES_NPY \
  --output PATH_TO_P3_OUTPUT/validation.query.json --queries

python -m p3.evaluate_fixed_disk \
  --index PATH_TO_P3_OUTPUT/adaptive_t0.49723.diskindex \
  --queries PATH_TO_P3_OUTPUT/validation.query.json \
  --qrels PATH_TO_VALIDATION_QRELS \
  --checkpoint PATH_TO_P3_OUTPUT/adaptive.validation.checkpoint.json \
  --output PATH_TO_P3_OUTPUT/adaptive.validation.json
```

Use `evaluate_fixed128_disk.py` for the dedicated k=128 index. Evaluators save
progress checkpoints and report MRR@10, Recall@1000, NDCG@10, index size, mean
and median query latency, and average passage concepts. Keep the query set,
host, scoring constants, and index implementation fixed for fair latency
comparisons.

## Recorded full-corpus validation result

The measured results and their interpretation are in
[`docs/P3_results.md`](../docs/P3_results.md). The adaptive configuration
evaluated there missed the planned quality and index-size target; report it as
a negative result. The binary indexes, caches, checkpoints, and logs are run
artifacts and should be generated locally or on the cluster rather than
committed to the repository.

## Upstream source

The official CL-SR implementation is pinned under
`upstream/Decoding-Dense-Embeddings`; initialize the submodule after cloning.
