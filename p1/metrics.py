"""Evaluate fixed candidate sets against independent TREC qrels."""
import math

import numpy as np

METRICS = ('mrr_at_10', 'ndcg_at_10', 'recall_at_1000')
# exponential: 2^r - 1 (default, P1's original); linear: r, as trec_eval / upstream cl-sr/evaluate.py.
GAINS = {'exponential': lambda r: 2 ** max(0, r) - 1, 'linear': lambda r: max(0, r)}


def read_qrels(path):
    qrels = {}
    with open(path) as source:
        for line in source:
            if not line.strip():
                continue
            qid, _, docid, relevance = line.split()
            qrels.setdefault(qid, {})[docid] = int(relevance)
    if not qrels:
        raise ValueError('Empty qrels')
    return qrels


def per_query(run, qrels, gain='exponential'):
    """Metrics for every qrels query with a positive label; a query absent from the run scores zero."""
    if gain not in GAINS:
        raise ValueError(f'Unknown gain {gain!r}; choose from {sorted(GAINS)}')
    to_gain = GAINS[gain]

    def dcg(grades):
        return sum(to_gain(r) / math.log2(i + 2) for i, r in enumerate(grades))

    scores = {}
    for qid, judgments in qrels.items():
        relevant = {d for d, r in judgments.items() if r > 0}
        if not relevant:
            continue
        ranking = run.get(qid, [])
        if len(ranking) != len(set(ranking)):
            raise ValueError(f'{qid}: duplicate ranked document')
        ideal = dcg(sorted(judgments.values(), reverse=True)[:10])
        scores[qid] = {
            'mrr_at_10': next((1 / i for i, d in enumerate(ranking[:10], 1) if d in relevant), 0.),
            'ndcg_at_10': dcg([judgments.get(d, 0) for d in ranking[:10]]) / ideal,
            'recall_at_1000': len(set(ranking[:1000]) & relevant) / len(relevant)}
    return scores


def evaluate(run, qrels, gain='exponential'):
    scores = per_query(run, qrels, gain)
    if not scores:
        raise ValueError('No queries with positive relevance judgments')
    return {**{k: sum(s[k] for s in scores.values()) / len(scores) for k in METRICS},
            'num_queries': len(scores)}


def randomization_test(a, b, permutations=10_000, seed=0, chunk=1000):
    """Two-sided paired approximate randomisation over per-query scores (sign flips of differences)."""
    if set(a) != set(b) or not a:
        raise ValueError('Paired test needs the same non-empty query set in both runs')
    qids = sorted(a)
    diff = np.array([b[q] - a[q] for q in qids])
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, permutations, chunk):
        signs = rng.choice((-1., 1.), size=(min(chunk, permutations - start), len(diff)))
        extreme += int((np.abs(signs @ diff / len(diff)) >= observed - 1e-12).sum())
    return (extreme + 1) / (permutations + 1)


def holm(p_values):
    """Holm step-down adjusted p-values, in input order; adjusted < alpha controls family-wise error at alpha."""
    m = len(p_values)
    adjusted, running = [0.] * m, 0.
    for rank, i in enumerate(sorted(range(m), key=lambda i: p_values[i])):
        running = max(running, min(1., (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted
