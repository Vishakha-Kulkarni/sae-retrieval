"""Shared evaluation harness: python -m p5.harness --help

python -m p5.harness compare --qrels val.qrels --run-a b0.run --run-b p5.run [--subset mismatch_1000.qids]

Runs may be upstream latent_retrieve.py output (qid docid rank score) or TREC (qid Q0 docid rank score tag).
Metrics use p1.metrics; every qrels query counts, and a query missing from a run scores zero.
"""
import argparse
import json
from pathlib import Path

from p1.metrics import METRICS, evaluate, per_query, randomization_test, read_qrels


def read_run(path):
    """Ranked docids per query, ordered by rank."""
    ranks = {}
    with open(path) as source:
        for line in source:
            fields = line.split()
            if not fields:
                continue
            if len(fields) == 4:
                qid, docid, rank, _ = fields
            elif len(fields) == 6:
                qid, _, docid, rank, _, _ = fields
            else:
                raise ValueError(f'{path}: run lines need 4 or 6 columns')
            rank = int(rank)
            if rank < 1:
                raise ValueError(f'{path}: ranks start at 1')
            docs = ranks.setdefault(qid, {})
            if docid in docs:
                raise ValueError(f'{path}: {qid} ranks {docid} twice')
            docs[docid] = rank
    return {qid: sorted(docs, key=docs.get) for qid, docs in ranks.items()}


def read_qids(path):
    qids = Path(path).read_text().split()
    if len(qids) != len(set(qids)):
        raise ValueError(f'{path}: duplicate query IDs')
    return qids


def restrict(qrels, qids):
    missing = set(qids) - qrels.keys()
    if missing:
        raise ValueError(f'{len(missing)} subset queries are not in the qrels, e.g. {sorted(missing)[0]}')
    return {q: qrels[q] for q in qids}


def compare(qrels_path, run_a, run_b, subset=None, gain='exponential', permutations=10_000, seed=0):
    qrels = read_qrels(qrels_path)
    if subset:
        qrels = restrict(qrels, read_qids(subset))
    runs = read_run(run_a), read_run(run_b)
    scores = [per_query(run, qrels, gain) for run in runs]
    return {'a': evaluate(runs[0], qrels, gain), 'b': evaluate(runs[1], qrels, gain),
            'p_value': {m: randomization_test({q: s[m] for q, s in scores[0].items()},
                                              {q: s[m] for q, s in scores[1].items()}, permutations, seed)
                        for m in METRICS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    cmp = commands.add_parser('compare', help='Metrics for two runs and paired randomisation p-values')
    cmp.add_argument('--qrels', required=True)
    cmp.add_argument('--run-a', required=True, help='Reference run, e.g. B0')
    cmp.add_argument('--run-b', required=True)
    cmp.add_argument('--subset', help='File of query IDs, e.g. a mismatch subset')
    cmp.add_argument('--gain', choices=['exponential', 'linear'], default='exponential',
                     help='NDCG gain; linear matches upstream cl-sr/evaluate.py on graded qrels')
    cmp.add_argument('--permutations', type=int, default=10_000)
    cmp.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(compare(args.qrels, args.run_a, args.run_b, args.subset, args.gain,
                             args.permutations, args.seed), indent=2))


if __name__ == '__main__':
    main()
