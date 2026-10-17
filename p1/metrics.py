"""Evaluate fixed candidate sets against independent TREC qrels."""
import math


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


def evaluate(run, qrels):
    totals = dict(mrr_at_10=0., ndcg_at_10=0., recall_at_1000=0.)
    count = 0
    for qid, judgments in qrels.items():
        relevant = {d for d, r in judgments.items() if r > 0}
        if not relevant:
            continue
        count += 1
        ranking = run.get(qid, [])
        if len(ranking) != len(set(ranking)):
            raise ValueError(f'{qid}: duplicate ranked document')
        totals['mrr_at_10'] += next((1 / i for i, d in enumerate(ranking[:10], 1)
                                          if d in relevant), 0.)
        def dcg(grades):
            return sum((2 ** max(0, r) - 1) / math.log2(i + 2)
                       for i, r in enumerate(grades))
        ideal = dcg(sorted(judgments.values(), reverse=True)[:10])
        totals['ndcg_at_10'] += dcg([judgments.get(d, 0) for d in ranking[:10]]) / ideal
        totals['recall_at_1000'] += len(set(ranking[:1000]) & relevant) / len(relevant)
    if not count:
        raise ValueError('No queries with positive relevance judgments')
    return {**{k: v / count for k, v in totals.items()}, 'num_queries': count}
