"""Vectorised copy of upstream cl-sr/latent_retrieve.py retrieve_one over the indexing.py pickle.

Upstream scores every posting in a Python loop, which takes seconds per query on the full MS MARCO index (a
few concepts fire on most passages). This keeps its float64 operations in the same order, its TOP_K and its
tie order (a stable sort over first-touch order), so rankings and scores are identical. The arrays are cached
next to the pickle as <index>.fast.npz and rebuilt when the pickle changes.
"""
import os
from pathlib import Path
import pickle

import numpy as np

from .upstream import upstream_functions

_UPSTREAM = upstream_functions('latent_retrieve.py')
K1, K2, B, TOP_K = (_UPSTREAM[name] for name in ('k1', 'k2', 'b', 'TOP_K'))


def load_pickle(path):
    with open(path, 'rb') as source:
        return pickle.load(source)


class FastIndex:
    def __init__(self, terms, offsets, rows, h_d, idf, doc_keys):
        self.position = {t: i for i, t in enumerate(terms.tolist())}
        self.offsets, self.rows, self.h_d, self.idf, self.doc_keys = offsets, rows, h_d, idf, doc_keys
        self.scores = np.zeros(len(doc_keys))
        self.first = np.full(len(doc_keys), -1, dtype=np.int64)

    @classmethod
    def from_index(cls, index):
        inverted, lengths, avgdl = index['inverted_index'], index['document_lengths'], index['avgdl']
        doc_keys = np.array(sorted(lengths), dtype=np.int64)
        length_of_row = np.array([lengths[d] for d in doc_keys.tolist()], dtype=np.int64)
        terms = list(inverted)
        sizes = [len(inverted[t]['postings']) for t in terms]
        offsets = np.zeros(len(terms) + 1, dtype=np.int64)
        np.cumsum(sizes, out=offsets[1:])
        rows = np.empty(offsets[-1], dtype=np.int64)
        tf = np.empty(offsets[-1])
        for i, t in enumerate(terms):
            postings = np.array(inverted[t]['postings'], dtype=np.float64)  # docids < 2**53 stay exact
            rows[offsets[i]:offsets[i + 1]] = np.searchsorted(doc_keys, postings[:, 0].astype(np.int64))
            tf[offsets[i]:offsets[i + 1]] = postings[:, 1]
        # Same expression and evaluation order as upstream bm25_score's h_d.
        h_d = (tf * (K1 + 1.0)) / (tf + K1 * (1 - B + B * (length_of_row[rows] / avgdl)))
        idf = np.array([inverted[t]['idf'] for t in terms])
        return cls(np.array(terms), offsets, rows.astype(np.int32), h_d, idf, doc_keys)

    @classmethod
    def load(cls, index_path):
        """Cached arrays for the pickle at index_path; built (and cached) when missing or stale."""
        index_path = Path(index_path)
        cache = index_path.with_name(index_path.name + '.fast.npz')
        stamp = np.array([index_path.stat().st_size, index_path.stat().st_mtime_ns], dtype=np.int64)
        if cache.exists():
            with np.load(cache) as data:
                if np.array_equal(data['source'], stamp):
                    return cls(*(data[k] for k in ('terms', 'offsets', 'rows', 'h_d', 'idf', 'doc_keys')))
        fast = cls.from_index(load_pickle(index_path))
        partial = cache.with_name(f'{cache.name}.{os.getpid()}.partial')
        with open(partial, 'wb') as target:
            np.savez(target, terms=np.array(list(fast.position)), offsets=fast.offsets, rows=fast.rows,
                     h_d=fast.h_d, idf=fast.idf, doc_keys=fast.doc_keys, source=stamp)
        os.replace(partial, cache)
        return fast

    def retrieve(self, terms, weights):
        """Upstream retrieve_one(terms, weights, ...): top-TOP_K (docid, score), and the postings scanned."""
        touched, base = [], 0
        for t, tf_q in zip(terms, weights):
            i = self.position.get(t)
            if i is None:
                continue  # upstream skips terms absent from the index
            rows = self.rows[self.offsets[i]:self.offsets[i + 1]]
            h_q = (tf_q * (K2 + 1.0)) / (tf_q + K2)
            self.scores[rows] += self.idf[i] * h_q * self.h_d[self.offsets[i]:self.offsets[i + 1]]
            new = self.first[rows] < 0
            self.first[rows[new]] = base + np.flatnonzero(new)
            touched.append(rows[new])
            base += len(rows)
        if not touched:
            return [], 0
        touched = np.concatenate(touched)  # first-touch order, as upstream's dict insertion order
        scores = self.scores[touched]
        if len(scores) > TOP_K:  # keep everything tied with the TOP_K-th score, in first-touch order
            kth = np.partition(scores, len(scores) - TOP_K)[len(scores) - TOP_K]
            keep = np.flatnonzero(scores >= kth)
        else:
            keep = np.arange(len(scores))
        order = keep[np.argsort(-scores[keep], kind='stable')][:TOP_K]
        ranked = [(int(d), float(s)) for d, s in zip(self.doc_keys[touched[order]], scores[order])]
        self.scores[touched] = 0.
        self.first[touched] = -1
        return ranked, base
