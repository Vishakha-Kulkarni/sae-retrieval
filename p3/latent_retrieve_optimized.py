from collections import defaultdict
import heapq
import latent_retrieve as original

TOP_K = original.TOP_K


def retrieve_one(query_terms, query_tfs, inv_idx, doc_lens, avgdl):
    """Optimized BM25 retrieval, preserving the original scoring formula."""
    scores = defaultdict(float)

    for t, tf_q in zip(query_terms, query_tfs):
        entry = inv_idx.get(t)
        if entry is None:
            continue

        idf = entry["idf"]
        h_q = (tf_q * (original.k2 + 1.0)) / (tf_q + original.k2)

        for docid, tf_d in entry["postings"]:
            doc_len = doc_lens[docid]
            h_d = (tf_d * (original.k1 + 1.0)) / (
                tf_d + original.k1 * (1 - original.b + original.b * (doc_len / avgdl))
            )
            scores[docid] += idf * h_q * h_d

    return heapq.nlargest(TOP_K, scores.items(), key=lambda x: x[1])
