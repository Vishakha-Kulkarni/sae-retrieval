"""Shared evaluation harness: python -m p5.harness --help

python -m p5.harness compare --qrels val.qrels --run-a b0.run --run-b p5.run [--subset mismatch_1000.qids]
python -m p5.harness family --qrels val.qrels --subset validation.mismatch_1000.qids --run L=L.val.run \
    --against B0=B0.val.run T=T0.7.val.run N=N8.val.run --output family.val.json

family is P5's formal claim: the run against each reference on Recall@1000 (all queries) and MRR@10
(mismatch@1000), with Holm's correction over all those tests. Other p-values (compare, row JSONs) are
uncorrected.

Runs may be upstream latent_retrieve.py output (qid docid rank score) or TREC (qid Q0 docid rank score tag).
Metrics use p1.metrics; every qrels query counts, and a query missing from a run scores zero.
"""
import argparse
import json
from pathlib import Path

from p1.metrics import METRICS, evaluate, holm, per_query, randomization_test, read_qrels

# The success criterion's metrics: the only cells P5 claims with a multiple-comparison correction.
PRIMARY = (('all', 'recall_at_1000'), ('mismatch_1000', 'mrr_at_10'))


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


def primary_family(qrels_path, subset_path, run, against, gain='exponential', permutations=10_000, seed=0,
                   alpha=.05):
    """P5's formal claims: one run against each reference on PRIMARY, Holm-corrected as one family.

    run is (name, path); against maps reference names (B0, T, N) to run paths.
    """
    qrels = read_qrels(qrels_path)
    views = {'all': qrels, 'mismatch_1000': restrict(qrels, read_qids(subset_path))}
    name, ours = run[0], read_run(run[1])
    tests = []
    for reference, path in against.items():
        theirs = read_run(path)
        for view, metric in PRIMARY:
            a, b = ({q: s[metric] for q, s in per_query(r, views[view], gain).items()} for r in (theirs, ours))
            tests.append({'comparison': f'{name} vs {reference}', 'view': view, 'metric': metric,
                          'num_queries': len(a), 'ours': sum(b.values()) / len(b),
                          'reference': sum(a.values()) / len(a),
                          'p': randomization_test(a, b, permutations, seed)})
    for test, adjusted in zip(tests, holm([t['p'] for t in tests])):
        test['p_holm'], test['significant'] = adjusted, adjusted < alpha
    return {'correction': 'holm', 'alpha': alpha, 'run': str(run[1]), 'qrels': str(qrels_path),
            'subset': str(subset_path), 'against': {k: str(v) for k, v in against.items()}, 'tests': tests}


def filter_to_subset(qrels_path, queries_path, size, prefix):
    """Smoke tier: keep judgments on passages 0..size-1 and the queries with a positive one among them.

    Writes <prefix>.qrels and <prefix>.queries.tsv; returns the kept query IDs in qrels order.
    """
    outputs = Path(f'{prefix}.qrels'), Path(f'{prefix}.queries.tsv')
    for path in outputs:
        if path.exists():
            raise FileExistsError(f'{path} exists; choose a new prefix')
    lines, kept = [], {}
    for line in Path(qrels_path).read_text().splitlines():
        qid, _, docid, relevance = line.split()
        if int(docid) < size:
            lines.append((qid, line))
            if int(relevance) > 0:
                kept.setdefault(qid, None)
    text = dict(line.split('\t', 1) for line in Path(queries_path).read_text().splitlines() if line)
    missing = kept.keys() - text.keys()
    if missing:
        raise ValueError(f'{len(missing)} kept queries have no text, e.g. {sorted(missing)[0]}')
    outputs[0].write_text(''.join(line + '\n' for qid, line in lines if qid in kept))
    outputs[1].write_text(''.join(f'{qid}\t{text[qid]}\n' for qid in kept))
    return list(kept)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    smoke = commands.add_parser('filter', help='Smoke tier: restrict qrels and queries to the first N passages')
    smoke.add_argument('--qrels', required=True)
    smoke.add_argument('--queries', required=True, help='MS MARCO queries TSV (qid<TAB>text)')
    smoke.add_argument('--size', type=int, required=True, help='Passages 0..size-1 are encoded and indexed')
    smoke.add_argument('--prefix', required=True, help='Writes <prefix>.qrels and <prefix>.queries.tsv')
    cmp = commands.add_parser('compare', help='Metrics for two runs and paired randomisation p-values')
    cmp.add_argument('--qrels', required=True)
    cmp.add_argument('--run-a', required=True, help='Reference run, e.g. B0')
    cmp.add_argument('--run-b', required=True)
    cmp.add_argument('--subset', help='File of query IDs, e.g. a mismatch subset')
    cmp.add_argument('--gain', choices=['exponential', 'linear'], default='exponential',
                     help='NDCG gain; linear matches upstream cl-sr/evaluate.py on graded qrels')
    cmp.add_argument('--permutations', type=int, default=10_000)
    cmp.add_argument('--seed', type=int, default=0)
    family = commands.add_parser('family', help='Primary metrics against several references, Holm-corrected')
    family.add_argument('--qrels', required=True)
    family.add_argument('--subset', required=True, help='Mismatch@1000 query IDs for the same split')
    family.add_argument('--run', required=True, metavar='NAME=RUN', help='The method row, e.g. L=L.val.run')
    family.add_argument('--against', nargs='+', required=True, metavar='NAME=RUN',
                        help='Reference rows, e.g. B0=B0.val.run T=T0.7.val.run N=N8.val.run')
    family.add_argument('--output', required=True, help='Result JSON')
    family.add_argument('--gain', choices=['exponential', 'linear'], default='linear')
    family.add_argument('--permutations', type=int, default=10_000)
    family.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'family':
        result = primary_family(args.qrels, args.subset, tuple(args.run.split('=', 1)),
                                dict(a.split('=', 1) for a in args.against), args.gain, args.permutations, args.seed)
        with open(args.output, 'x') as target:
            json.dump(result, target, indent=2)
        for t in result['tests']:
            print(f"{t['comparison']:<12} {t['view']:<14} {t['metric']:<15} {t['ours']:.4f} vs {t['reference']:.4f}"
                  f"  p={t['p']:.4f}  holm={t['p_holm']:.4f}{'  *' if t['significant'] else ''}")
        return
    if args.command == 'filter':
        kept = filter_to_subset(args.qrels, args.queries, args.size, args.prefix)
        print(f'kept {len(kept)} queries')
        return
    print(json.dumps(compare(args.qrels, args.run_a, args.run_b, args.subset, args.gain,
                             args.permutations, args.seed), indent=2))


if __name__ == '__main__':
    main()
