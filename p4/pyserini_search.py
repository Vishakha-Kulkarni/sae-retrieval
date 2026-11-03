"""P4: CL-SR search over the Pyserini impact index, with per-query latency and optional evaluation.

python -m p4.pyserini_search --index pyserini/ --queries dl19.query.json --output runs/dl19.trec \
    --qrels trec_dl2019_qrels.txt

The index stores the passage-side BM25 factor h_d (p4.pyserini_index). Here each query concept gets
idf * h_q, so Lucene's impact score sum(q_w * d_w) equals the CL-SR score up to quantization.
Latency follows the paper (Appendix H): batch size 1, all queries, best of --repeats full runs.
Only the Lucene search call is timed; the index is loaded and warmed up first.
"""
import argparse
import json
import os
import math
from pathlib import Path
import time

import numpy as np

K2 = float(os.environ.get("K2", 1.5))  


def idf(df, num_docs, kind):
    if kind == 'repo':  # released cl-sr/indexing.py; negative for concepts in > half of passages
        return math.log((num_docs - df + 0.5) / (df + 0.5))
    return math.log(num_docs / (1 + df))  # paper Eq. 6


def query_weights(record, df, num_docs, kind, scale, k2=K2):
    """{concept: integer weight} for one query; concepts unseen in the index or with idf <= 0 are dropped
    (Lucene impacts must be positive). Returns the weights and how many concepts were dropped."""
    out, dropped = {}, 0
    for concept, tf in zip(record['ids'], record['weight']):
        c = int(concept)
        if c >= len(df) or df[c] == 0:
            continue  # no passage has it: contributes nothing in the released code either
        weight = idf(int(df[c]), num_docs, kind) * tf * (k2 + 1.0) / (tf + k2)
        value = int(round(scale * weight))
        if value < 1:
            dropped += 1
            continue
        out[str(c)] = out.get(str(c), 0) + value
    return out, dropped


def load_searcher(index_dir):
    from pyserini.pyclass import autoclass
    return autoclass('io.anserini.search.SimpleImpactSearcher')(str(index_dir))


def to_java(weights):
    from pyserini.pyclass import JHashMap, JInt
    query = JHashMap()
    for token, value in weights.items():
        query.put(token, JInt(value))
    return query


def search_all(searcher, queries, k, repeats):
    """Returns {qid: [(docid, raw_score), ...]} and per-query latencies (ms) from the fastest run."""
    for q in list(queries.values())[:20]:  # warm-up: JIT compilation and OS page cache
        searcher.search(q, k)
    best, results = None, {}
    for _ in range(repeats):
        times = []
        for qid, q in queries.items():
            start = time.perf_counter()
            hits = searcher.search(q, k)
            times.append((time.perf_counter() - start) * 1000)
            results[qid] = [(h.docid, h.score) for h in hits]
        if best is None or sum(times) < sum(best):
            best = times
    return results, np.array(best)


def evaluate(results, qrels_path):
    import pytrec_eval
    qrels = {}
    for line in open(qrels_path):
        qid, _, pid, rel = line.split()
        qrels.setdefault(qid, {})[pid] = int(rel)
    run = {q: {d: float(s) for d, s in hits} for q, hits in results.items() if q in qrels}
    top10 = {q: {d: float(s) for d, s in hits[:10]} for q, hits in results.items() if q in qrels}
    scores = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.10', 'recall.1000'}).evaluate(run)
    rr = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank'}).evaluate(top10)
    mean = lambda d, m: round(sum(v[m] for v in d.values()) / max(len(d), 1), 5)
    return {'NDCG@10': mean(scores, 'ndcg_cut_10'), 'MRR@10': mean(rr, 'recip_rank'),
            'Recall@1000': mean(scores, 'recall_1000'), 'evaluated_queries': len(scores)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--index', required=True, help='Output directory of p4.pyserini_index')
    parser.add_argument('--queries', required=True, help='{qid: {"ids": [...], "weight": [...]}} JSON')
    parser.add_argument('--output', required=True, help='TREC run file')
    parser.add_argument('--k', type=int, default=1000)
    parser.add_argument('--idf', choices=['repo', 'paper'], default='repo')
    parser.add_argument('--query-scale', type=float, default=100.)
    parser.add_argument('--repeats', type=int, default=3, help='Paper: best of 3 full runs')
    parser.add_argument('--qrels', help='If given, report NDCG@10 / MRR@10 / Recall@1000')
    parser.add_argument('--tag', default='p4-clsr')
    args = parser.parse_args()

    root = Path(args.index)
    meta = json.loads((root / 'meta.json').read_text())
    df = np.load(root / 'df.npy')
    raw = json.loads(Path(args.queries).read_text())
    queries, dropped, empty = {}, 0, 0
    for qid, record in raw.items():
        weights, d = query_weights(record, df, meta['num_docs'], args.idf, args.query_scale)
        dropped += d
        empty += not weights
        queries[qid] = to_java(weights)

    load_start = time.perf_counter()
    searcher = load_searcher(root / 'index')
    load_s = time.perf_counter() - load_start
    results, latency = search_all(searcher, queries, args.k, args.repeats)

    scale = meta['scale'] * args.query_scale
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as run:
        for qid, hits in results.items():
            for rank, (docid, score) in enumerate(hits, 1):
                run.write(f'{qid} Q0 {docid} {rank} {score / scale:.4f} {args.tag}\n')

    report = {'queries': len(queries), 'empty_queries': empty, 'concepts_dropped_idf_le_0': dropped,
              'idf': args.idf, 'k': args.k, 'index_gb': meta.get('index_gb'), 'index_load_s': round(load_s, 2),
              'latency_ms': {'mean': round(float(latency.mean()), 2), 'median': round(float(np.median(latency)), 2),
                             'p95': round(float(np.percentile(latency, 95)), 2),
                             'total_s': round(float(latency.sum()) / 1000, 2)},
              'queries_per_s': round(len(latency) / (latency.sum() / 1000), 1)}
    if args.qrels:
        report['metrics'] = evaluate(results, args.qrels)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()