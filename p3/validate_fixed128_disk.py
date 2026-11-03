import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLSR = ROOT / "upstream" / "Decoding-Dense-Embeddings" / "cl-sr"
sys.path.insert(0, str(CLSR))

from p3.fixed128_disk import Fixed128DiskIndex
import latent_retrieve_optimized as upstream

P3_OUT = ROOT / "sae-simlm-k32-baseline/data_titled/p3_full_eval"
INDEX_PATH = P3_OUT / "fixed128.diskindex"
QUERY_PATH = P3_OUT / "validation.query.json"

N_DOCS = 100_000
N_QUERIES = 3
TOP_K = 1000

upstream.TOP_K = TOP_K

print("Opening disk index...", flush=True)
full = Fixed128DiskIndex(INDEX_PATH)

with open(QUERY_PATH) as f:
    queries = json.load(f)

subset = object.__new__(Fixed128DiskIndex)
subset.total_docs = N_DOCS
subset.avgdl = full.avgdl
subset.latent_dim = full.latent_dim
subset.shard_count = full.shard_count
subset.terms_per_shard = full.terms_per_shard
subset.doc_lengths = full.doc_lengths[:N_DOCS]
subset.term_stats = full.term_stats
subset._shards = {}

all_passed = True

for qnum, (qid, values) in enumerate(list(queries.items())[:N_QUERIES], 1):
    print(f"\nQuery {qnum}/{N_QUERIES}: {qid}", flush=True)

    terms = [int(x) for x in values["ids"]]
    weights = values["weight"]

    if len(terms) != len(weights):
        raise ValueError(f"Query {qid}: IDs and weights differ")

    upstream_index = {}
    shard_records = {}

    for term in terms:
        if term < 0 or term >= full.latent_dim:
            continue

        stats = full.term_stats[term]
        if int(stats["doc_freq"]) == 0:
            continue

        shard_id = term // full.terms_per_shard
        records = full._get_shard(shard_id)

        if records is None:
            continue

        term_values = records["term"]
        lo = int(np.searchsorted(term_values, term, side="left"))
        hi = int(np.searchsorted(term_values, term, side="right"))

        postings = records[lo:hi]
        postings = postings[postings["docid"] < N_DOCS]

        idf = float(stats["idf"])

        upstream_index[str(term)] = {
            "idf": idf,
            "postings": [
                (int(row["docid"]), float(row["weight"]))
                for row in postings
            ],
        }

        if len(postings):
            shard_records.setdefault(shard_id, []).append(
                postings.copy()
            )

    # Prepare sorted, subset-only shards for the disk retriever.
    for shard_id, arrays in shard_records.items():
        combined = np.concatenate(arrays)
        combined.sort(order=["term", "docid"], kind="quicksort")
        subset._shards[shard_id] = combined

    # Reference: upstream Python BM25 implementation.
    reference = upstream.retrieve_one(
        [str(x) for x in terms],
        weights,
        upstream_index,
        {i: int(full.doc_lengths[i]) for i in range(N_DOCS)},
        full.avgdl,
    )

    # Candidate: new disk-backed NumPy implementation.
    actual = subset.retrieve_one(terms, weights, top_k=TOP_K)

    ref_ids = [int(docid) for docid, _ in reference]
    actual_ids = [int(docid) for docid, _ in actual]

    ref_scores = dict(reference)
    actual_scores = dict(actual)

    common = set(ref_scores) & set(actual_scores)
    overlap = len(common)
    max_abs_diff = max(
        (abs(ref_scores[d] - actual_scores[d]) for d in common),
        default=0.0,
    )

    same_ranking = ref_ids == actual_ids
    scores_close = all(
        np.isclose(
            ref_scores[d],
            actual_scores[d],
            rtol=1e-6,
            atol=1e-8,
        )
        for d in common
    )

    passed = same_ranking and scores_close
    all_passed &= passed

    print(f"Reference results: {len(ref_ids)}", flush=True)
    print(f"Disk results:      {len(actual_ids)}", flush=True)
    print(f"Top-k overlap:     {overlap}/{min(len(ref_ids), len(actual_ids))}")
    print(f"Same ranking:      {same_ranking}")
    print(f"Scores close:      {scores_close}")
    print(f"Max score diff:    {max_abs_diff:.12g}")
    print(f"Result:            {'PASS' if passed else 'CHECK'}", flush=True)

    # Also exercise the actual full-corpus retrieval path.
    full_hits = full.retrieve_one(terms, weights, top_k=TOP_K)
    print(f"Full-corpus hits:  {len(full_hits)}", flush=True)

print("\n" + ("ALL TESTS PASSED" if all_passed else "SOME TESTS NEED INVESTIGATION"))
sys.exit(0 if all_passed else 1)
