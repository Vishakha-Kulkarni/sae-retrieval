"""Released CL-SR scoring, vectorised: python -m p2.clsr --help

Computes exactly the score of upstream cl-sr/latent_retrieve.py over the index built by
cl-sr/indexing.py (same k1/k2/b, same IDF, same per-passage truncation to --max-latents, same
length normalisation), as a sparse matrix product instead of a Python loop over postings.
Upstream needs ~16 s per query on 8.8M passages; this takes well under a second.
--check-run compares scores with an (even partial) upstream run file, so parity is verified on
real data rather than assumed. Evaluate the output with upstream cl-sr/evaluate.py as before.
"""
import argparse
import json

import numpy as np
from scipy import sparse

K1, K2, B = 0.6, 1.5, 2.0  # upstream latent_retrieve.py


def load_passages(path, max_latents=None, num_concepts=None):
    """Read upstream passage latents with indexing.py's truncation: top max_latents by weight."""
    docids, indptr, cols, vals = [], [0], [], []
    with open(path) as source:
        for line in source:
            record = json.loads(line)
            pairs = sorted(zip(record['ids'], record['weight']), key=lambda p: p[1], reverse=True)
            if max_latents is not None:
                pairs = pairs[:max_latents]
            docids.append(record['docid'])
            cols.extend(int(c) for c, _ in pairs)
            vals.extend(float(w) for _, w in pairs)
            indptr.append(len(cols))
    width = num_concepts or (max(cols) + 1)
    tf = sparse.csr_matrix((np.asarray(vals, np.float32), np.asarray(cols, np.int32),
                            np.asarray(indptr, np.int64)), shape=(len(docids), width))
    return docids, tf


def passage_impacts(tf):
    """idf(t) * h_d for every posting, as in upstream bm25_score (doc length = concepts kept)."""
    lengths = np.diff(tf.indptr).astype(np.float64)
    avgdl = lengths.mean()
    df = np.bincount(tf.indices, minlength=tf.shape[1]).astype(np.float64)
    n = tf.shape[0]
    idf = np.log((n - df + 0.5) / (df + 0.5))
    row_norm = np.repeat(K1 * (1 - B + B * lengths / avgdl), np.diff(tf.indptr))
    data = tf.data.astype(np.float64)
    impact = idf[tf.indices] * data * (K1 + 1) / (data + row_norm)
    out = sparse.csr_matrix((impact, tf.indices, tf.indptr), shape=tf.shape)
    return out, df


def query_matrix(queries, width, df):
    """h_q per query term; terms absent from the index are skipped, as upstream does."""
    qids, indptr, cols, vals = [], [0], [], []
    for qid, record in queries.items():
        for c, w in zip(record['ids'], record['weight']):
            c = int(c)
            if c < width and df[c] > 0:
                cols.append(c)
                vals.append(float(w) * (K2 + 1) / (float(w) + K2))
        qids.append(qid)
        indptr.append(len(cols))
    return qids, sparse.csr_matrix((vals, cols, indptr), shape=(len(qids), width))


def search(queries_m, impacts_t, top_k=1000, batch=64):
    """Yield (row, docs, scores) for each query, highest score first. impacts_t: concepts x passages."""
    for start in range(0, queries_m.shape[0], batch):
        scores = (queries_m[start:start + batch] @ impacts_t).tocsr()
        for r in range(scores.shape[0]):
            row = scores.getrow(r)
            k = min(top_k, row.nnz)
            if k == 0:
                yield start + r, np.array([], np.int64), np.array([])
                continue
            top = np.argpartition(-row.data, k - 1)[:k]
            top = top[np.argsort(-row.data[top], kind='stable')]
            yield start + r, row.indices[top], row.data[top]


def check(run_path, ours, tolerance=1e-3):
    """Compare per (qid, docid) scores with an upstream run; upstream prints scores with 3 decimals."""
    upstream = {}
    with open(run_path) as source:
        for line in source:
            parts = line.split()
            if len(parts) == 4:
                upstream.setdefault(parts[0], {})[parts[1]] = float(parts[3])
    if upstream:  # a still-running upstream job may have written only part of its last query
        upstream.pop(list(upstream)[-1])
    compared = mismatched = 0
    top10_same = 0
    for qid, docs in upstream.items():
        mine = ours.get(qid)
        if mine is None:
            continue
        for docid, score in docs.items():
            compared += 1
            if docid not in mine or abs(mine[docid] - score) > tolerance:
                mismatched += 1
        up10 = sorted(docs, key=lambda d: -docs[d])[:10]
        my10 = sorted(mine, key=lambda d: -mine[d])[:10]
        top10_same += set(up10) == set(my10)
    return {'queries_compared': sum(1 for q in upstream if q in ours),
            'postings_compared': compared, 'mismatched': mismatched, 'same_top10_set': top10_same}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--passage-latents', required=True, help='Upstream-format passage JSONL')
    parser.add_argument('--query-latents', required=True, help='Upstream-format query JSON')
    parser.add_argument('--output', required=True, help='Run in latent_retrieve.py format: qid docid rank score')
    parser.add_argument('--max-latents', type=int, default=24,
                        help='indexing.py --max_latents (README: 24); 0 keeps every active concept')
    parser.add_argument('--top-k', type=int, default=1000)
    parser.add_argument('--check-run', help='Upstream latent_retrieve.py output (may be partial) to verify against')
    args = parser.parse_args()
    queries = json.load(open(args.query_latents))
    width = 1 + max([int(c) for q in queries.values() for c in q['ids']] or [0])
    docids, tf = load_passages(args.passage_latents, args.max_latents or None)
    width = max(width, tf.shape[1])
    tf.resize((tf.shape[0], width))
    impacts, df = passage_impacts(tf)
    impacts_t = impacts.T.tocsr()  # concepts x passages: one row per posting list
    qids, qm = query_matrix(queries, width, df)
    docids = np.asarray(docids)
    ours = {}
    with open(args.output, 'w') as out:
        for r, docs, scores in search(qm, impacts_t, args.top_k):
            qid = qids[r]
            for rank, (d, s) in enumerate(zip(docids[docs], scores), 1):
                out.write(f'{qid}\t{d}\t{rank}\t{s:.3f}\n')
            if args.check_run:
                ours[qid] = {str(d): float(s) for d, s in zip(docids[docs], scores)}
    if args.check_run:
        print(json.dumps({'parity_with_upstream': check(args.check_run, ours)}))


if __name__ == '__main__':
    main()
