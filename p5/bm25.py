"""Differentiable copy of the released CL-SR BM25 (cl-sr/latent_retrieve.py bm25_score), query side only.

score(q, d) = sum over shared concepts c of idf(c) * h_q(w_c) * h_d(f_dc, |d|), with upstream k1, k2, b,
BM25 IDF and |d| = indexed concept count. Passage terms idf * h_d are constants; only h_q(w) carries
gradient. stats must describe the index (p1.prepare corpus --max-concepts = indexing.py --max_latents).
"""
import torch

from p1.data import denominator, idf
from .upstream import upstream_functions

_UPSTREAM = upstream_functions('latent_retrieve.py')
K1, K2, B = _UPSTREAM['k1'], _UPSTREAM['k2'], _UPSTREAM['b']


def query_saturation(weights):
    return weights * (K2 + 1) / (weights + K2)


def doc_terms(candidates, columns, stats):
    """Sparse (row, column, idf * h_d) entries for candidate passages ({concept: weight}) over concept columns."""
    position = {c: j for j, c in enumerate(columns)}
    rows, cols, values = [], [], []
    for i, doc in enumerate(candidates):
        length = len(doc) / stats['avg_doc_length']
        for concept, tf in doc.items():
            j = position.get(concept)
            if j is not None:
                rows.append(i)
                cols.append(j)
                values.append(idf(concept, stats) * tf * (K1 + 1) / denominator(tf, length, K1, B))
    return (torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long),
            torch.tensor(values, dtype=torch.float32))


def scores(terms, weights, num_docs):
    """BM25 score per candidate for query weights over the doc_terms columns."""
    rows, cols, values = terms
    contributions = values * query_saturation(weights)[cols]
    return torch.zeros(num_docs, dtype=contributions.dtype).index_add(0, rows, contributions)
