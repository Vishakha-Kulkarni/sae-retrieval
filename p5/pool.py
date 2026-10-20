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


def eligible(stats, num_concepts):
    """Concepts an expansion may add: present in the index vocabulary with IDF > 0 (others lower scores)."""
    mask = np.zeros(num_concepts, dtype=bool)
    for concept in stats['document_frequency']:
        if concept < num_concepts and idf(concept, stats) > 0:
            mask[concept] = True
    return mask


def near_miss(pre, threshold, m):
    """Per row, the m concepts with the largest positive pre-activation at or below the threshold."""
    below = pre.masked_fill((pre > threshold) | (pre <= 0), float('-inf'))
    values, top = below.topk(min(m, pre.shape[1]), dim=1)
    return [[c for c, v in zip(row, vals) if v > float('-inf')] for row, vals in zip(top.tolist(), values.tolist())]
