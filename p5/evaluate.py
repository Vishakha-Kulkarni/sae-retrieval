"""Row evaluator: one query latent file + the shared CL-SR index -> one ablation-table row.

python -m p5.evaluate --name B0 --queries val.query.json --index index.pkl --qrels val.qrels \
    --subset mismatch_1000=val.mismatch_1000.qids --run-output B0.val.run --output B0.val.json

Retrieval is upstream cl-sr/latent_retrieve.py (retrieve_one, its BM25 constants and TOP_K), loaded unchanged.
Concept IDs are passed as strings because upstream indexing.py keys postings by str(term); upstream main()
passes the JSON's integer IDs, which match no posting. Metrics follow p5.harness: every qrels query counts.
Latency is single-threaded wall time of retrieve_one; postings touched is the hardware-independent cost.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

from p1.metrics import METRICS, evaluate, per_query, randomization_test, read_qrels
from .harness import read_qids, read_run, restrict
from .upstream import upstream_functions


def load_index(path):
    with open(path, 'rb') as source:
        return pickle.load(source)


def retrieve(query, index, upstream):
    """Upstream top-TOP_K (docid, score) list for one query, and the number of postings it scanned."""
    terms = [str(c) for c in query['ids']]
    postings = sum(len(index['inverted_index'][t]['postings']) for t in terms if t in index['inverted_index'])
    ranked = upstream['retrieve_one'](terms, query['weight'], index['inverted_index'],
                                      index['document_lengths'], index['avgdl'])
    return ranked, postings


def file_hash(path):
    with open(path, 'rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def evaluate_row(name, queries_path, index_path, qrels_path, subsets=None, baseline_run=None,
                 run_output=None, gain='exponential', permutations=10_000, seed=0):
    subsets = subsets or {}
    if run_output and Path(run_output).exists():
        raise FileExistsError(f'{run_output} exists; choose a new run path')
    queries = json.loads(Path(queries_path).read_text())
    qrels = read_qrels(qrels_path)
    missing = qrels.keys() - queries.keys()
    if missing:
        raise ValueError(f'{len(missing)} qrels queries have no query latents, e.g. {sorted(missing)[0]}')
    index, upstream = load_index(index_path), upstream_functions('latent_retrieve.py')
    run, latencies, postings = {}, [], []
    for qid in sorted(qrels):
        started = time.perf_counter()
        ranked, touched = retrieve(queries[qid], index, upstream)
        latencies.append((time.perf_counter() - started) * 1000)
        postings.append(touched)
        run[qid] = ranked
    if run_output:
        with open(run_output, 'x') as target:  # upstream latent_retrieve.py output format
            for qid, ranked in run.items():
                for rank, (docid, score) in enumerate(ranked, 1):
                    target.write(f'{qid}\t{docid}\t{rank}\t{score:.3f}\n')
    ranking = {qid: [str(d) for d, _ in ranked] for qid, ranked in run.items()}
    views = {'all': qrels, **{s: restrict(qrels, read_qids(p)) for s, p in subsets.items()}}
    row = {'row': name, 'queries': str(queries_path), 'index': str(index_path), 'qrels': str(qrels_path),
           'subsets': {s: str(p) for s, p in subsets.items()},
           'input_sha256': {'queries': file_hash(queries_path), 'qrels': file_hash(qrels_path)},
           'metrics': {view: evaluate(ranking, q, gain) for view, q in views.items()},
           'gain': gain,
           'avg_query_concepts': float(np.mean([len(queries[q]['ids']) for q in qrels])),
           'avg_postings_touched': float(np.mean(postings)),
           'latency_ms': {'mean': float(np.mean(latencies)), 'p50': float(np.percentile(latencies, 50)),
                          'p95': float(np.percentile(latencies, 95))},
           'index_size_bytes': Path(index_path).stat().st_size}
    if baseline_run:
        baseline = read_run(baseline_run)
        row['baseline_run'] = str(baseline_run)
        row['p_value'] = {}
        for view, q in views.items():
            ours, theirs = per_query(ranking, q, gain), per_query(baseline, q, gain)
            row['p_value'][view] = {m: randomization_test({k: s[m] for k, s in theirs.items()},
                                                          {k: s[m] for k, s in ours.items()}, permutations, seed)
                                    for m in METRICS}
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True, help='Row label, e.g. B0, T0.7, N8, L8')
    parser.add_argument('--queries', required=True, help='Query latent JSON (upstream format)')
    parser.add_argument('--index', required=True, help='Upstream indexing.py pickle')
    parser.add_argument('--qrels', required=True)
    parser.add_argument('--subset', action='append', default=[], metavar='NAME=QIDS',
                        help='Extra metric view over a query subset, e.g. mismatch_1000=val.mismatch_1000.qids')
    parser.add_argument('--baseline-run', help='Run to test against (paired randomisation), e.g. B0')
    parser.add_argument('--run-output', help='Write this row\'s run (upstream 4-column format)')
    parser.add_argument('--output', required=True, help='Row result JSON')
    parser.add_argument('--gain', choices=['exponential', 'linear'], default='exponential')
    parser.add_argument('--permutations', type=int, default=10_000)
    args = parser.parse_args()
    subsets = dict(s.split('=', 1) for s in args.subset)
    row = evaluate_row(args.name, args.queries, args.index, args.qrels, subsets, args.baseline_run,
                       args.run_output, args.gain, args.permutations)
    with open(args.output, 'x') as target:
        json.dump(row, target, indent=2)
    print(json.dumps({k: row[k] for k in ('row', 'metrics', 'avg_query_concepts', 'latency_ms')}, indent=2))


if __name__ == '__main__':
    main()
