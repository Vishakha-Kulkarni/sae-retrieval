"""Candidate concepts for query expansion: decoder neighbours, near-miss concepts and the KTD5 filter.

Decoder directions are the columns of linear2.weight [768 x m]. p2.sae normalises rows (upstream
convention), so columns are normalised here before cosine.
"""
import numpy as np
import torch

from p1.data import idf


def decoder_directions(model):
    """Unit-norm decoder direction per concept, [m, d]."""
    columns = model.linear2.weight.detach().float().T
    return columns / columns.norm(dim=1, keepdim=True).clamp(min=1e-12)


@torch.no_grad()
def neighbour_table(directions, n, chunk=1024):
    """Top-n other concepts by decoder cosine for every concept: (indices [m, n], cosines [m, n])."""
    indices, cosines = [], []
    for start in range(0, len(directions), chunk):
        similarity = directions[start:start + chunk] @ directions.T
        rows = torch.arange(similarity.shape[0])
        similarity[rows, rows + start] = float('-inf')  # a concept is not its own neighbour
        values, top = similarity.topk(min(n, len(directions) - 1), dim=1)
        indices.append(top)
        cosines.append(values)
    return torch.cat(indices).numpy(), torch.cat(cosines).numpy()


FEATURES = ('pre_activation', 'query_cosine', 'neighbour_score', 'idf', 'log_active_count')


def idf_vector(stats, num_concepts):
    """BM25 IDF per concept; NaN outside the index vocabulary."""
    values = np.full(num_concepts, np.nan)
    for concept in stats['document_frequency']:
        if concept >= num_concepts:
            raise ValueError(f'stats concept {concept} is outside the {num_concepts}-concept SAE dictionary')
        values[concept] = idf(concept, stats)
    return values


def eligible(stats, num_concepts):
    """Concepts an expansion may add: present in the index vocabulary with IDF > 0 (others lower scores)."""
    return idf_vector(stats, num_concepts) > 0


def near_miss(pre, threshold, m):
    """Per row, the m concepts with the largest positive pre-activation at or below the threshold."""
    below = pre.masked_fill((pre > threshold) | (pre <= 0), float('-inf'))
    values, top = below.topk(min(m, pre.shape[1]), dim=1)
    return [[c for c, v in zip(row, vals) if v > float('-inf')] for row, vals in zip(top.tolist(), values.tolist())]


def pool_context(sae, stats, per_concept):
    """(directions, neighbour indices, IDF per concept), shared by every query's build_pool."""
    directions = decoder_directions(sae)
    return directions, neighbour_table(directions, per_concept)[0], idf_vector(stats, len(directions))


def build_pool(query, pre, embedding, directions, neighbours, idf_values, threshold, near):
    """KTD5 expansion candidates for one query ({concept: weight}) and their KTD7 features, in FEATURES order.

    pre: SAE pre-activations [m]; embedding: SimLM vector [d]; neighbours: neighbour_table indices [m, N].
    """
    active = list(query)
    pool = set(near_miss(pre[None], threshold, near)[0]) | set(neighbours[active].ravel().tolist())
    concepts = np.array(sorted(c for c in pool if idf_values[c] > 0 and c not in query), dtype=np.int64)
    d = directions[concepts]
    query_cosine = d @ (embedding / embedding.norm().clamp(min=1e-12))
    if active:
        weights = torch.tensor(list(query.values()), dtype=d.dtype)
        neighbour = (d @ directions[active].T * weights).max(1).values
    else:
        neighbour = torch.zeros(len(concepts))
    features = torch.stack([pre[concepts], query_cosine, neighbour,
                            torch.as_tensor(idf_values[concepts], dtype=d.dtype),
                            torch.full((len(concepts),), float(np.log1p(len(active))))], 1)
    return concepts, features
