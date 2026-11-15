"""Evaluate fixed candidate sets against independent TREC qrels."""
import math
METRICS = ('mrr_at_10', 'ndcg_at_10', 'recall_at_1000')


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
    if gain not in ('linear', 'exponential'):
        raise ValueError('gain must be linear or exponential')
    scores = {}
    for qid, judgments in qrels.items():
        relevant = {d for d, r in judgments.items() if r > 0}
        ranking = run.get(qid, [])
        if len(ranking) != len(set(ranking)):
            raise ValueError(f'{qid}: duplicate ranked document')
        mrr = next((1 / i for i, d in enumerate(ranking[:10], 1) if d in relevant), 0.)
        def dcg(grades):
            return sum(((2 ** max(0, r) - 1) if gain == 'exponential' else max(0, r)) / math.log2(i + 2)
                       for i, r in enumerate(grades))
        ideal = dcg(sorted(judgments.values(), reverse=True)[:10])
        scores[qid] = {'mrr_at_10': mrr,
                       'ndcg_at_10': dcg([judgments.get(d, 0) for d in ranking[:10]]) / ideal if ideal else 0.,
                       'recall_at_1000': len(set(ranking[:1000]) & relevant) / len(relevant) if relevant else 0.}
    return scores


def evaluate(run, qrels, gain='exponential'):
    rows = per_query(run, qrels, gain)
    if not rows:
        raise ValueError('Empty evaluation query set')
    return {**{m: sum(r[m] for r in rows.values()) / len(rows) for m in METRICS}, 'num_queries': len(rows)}


def randomization_test(a, b, permutations=10000, seed=0):
    import numpy as np
    if set(a) != set(b) or not a or permutations < 1:
        raise ValueError('Paired tests need the same nonempty query IDs and positive permutation count')
    delta = np.array([b[q] - a[q] for q in sorted(a)], dtype=np.float64)
    if not np.isfinite(delta).all():
        raise ValueError('Nonfinite per-query scores')
    observed = abs(delta.mean())
    rng, extreme = np.random.default_rng(seed), 0
    for start in range(0, permutations, 256):
        signs = rng.integers(0, 2, size=(min(256, permutations - start), len(delta)), dtype=np.int8) * 2 - 1
        extreme += int(np.count_nonzero(np.abs((signs * delta).mean(1)) >= observed - 1e-15))
    return (extreme + 1) / (permutations + 1)


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, maximum = [0.] * len(order), 0.
    for rank, index in enumerate(order):
        maximum = max(maximum, min(1., (len(order) - rank) * pvalues[index]))
        adjusted[index] = maximum
    return adjusted
