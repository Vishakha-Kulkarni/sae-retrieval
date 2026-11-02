"""Paired final-run comparisons; run files may be TREC or the upstream four-column format."""
import argparse
import json
import math
from pathlib import Path

from .metrics import METRICS, evaluate, holm, per_query, randomization_test, read_qrels


def read_run(path):
    rows = {}
    with open(path) as source:
        for line in source:
            parts = line.split()
            if not parts:
                continue
            if len(parts) == 6:
                qid, _, docid, rank, score, _ = parts
            elif len(parts) == 4:
                qid, docid, rank, score = parts
            else:
                raise ValueError('Run lines need four or six columns')
            rank = int(rank)
            if rank < 1 or not math.isfinite(float(score)):
                raise ValueError('Invalid rank or score')
            query = rows.setdefault(qid, {})
            if rank in query:
                raise ValueError('Duplicate rank')
            query[rank] = docid
    return {q: [docs[r] for r in sorted(docs)] for q, docs in rows.items()}


def compare(qrels, learned, references, gain='linear', metric='mrr_at_10', permutations=10000):
    ours = per_query(learned, qrels, gain)
    tests = []
    for name, run in references.items():
        theirs = per_query(run, qrels, gain)
        p = randomization_test({q: r[metric] for q, r in theirs.items()},
                               {q: r[metric] for q, r in ours.items()}, permutations, 42)
        tests.append({'reference': name, 'metric': metric,
                      'difference': sum(ours[q][metric] - theirs[q][metric] for q in ours) / len(ours),
                      'p_value': p})
    for row, adjusted in zip(tests, holm([r['p_value'] for r in tests])):
        row.update(p_holm=adjusted, significant_at_05=adjusted < .05,
                   significant_improvement_at_05=adjusted < .05 and row['difference'] > 0)
    return {'primary_metric': metric, 'ndcg_gain': gain, 'permutations': permutations,
            'learned': evaluate(learned, qrels, gain),
            'references': {name: evaluate(run, qrels, gain) for name, run in references.items()},
            'paired_randomization_tests': tests,
            'scope': 'Inference depends on whether these queries were untouched during selection; do not claim held-out significance on validation'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qrels', required=True)
    p.add_argument('--learned', required=True)
    p.add_argument('--reference', action='append', required=True, help='NAME=run_path; repeat for each baseline')
    p.add_argument('--gain', choices=['linear', 'exponential'], default='linear')
    p.add_argument('--metric', choices=METRICS, default='mrr_at_10')
    p.add_argument('--permutations', type=int, default=10000)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    references = dict(item.split('=', 1) for item in a.reference)
    result = compare(read_qrels(a.qrels), read_run(a.learned),
                     {name: read_run(path) for name, path in references.items()},
                     a.gain, a.metric, a.permutations)
    Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
