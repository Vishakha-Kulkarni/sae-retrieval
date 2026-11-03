from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RECORD_DTYPE = np.dtype([
    ("term", "<u2"),
    ("docid", "<u4"),
    ("weight", "<f4"),
])

K1 = 0.6
K2 = 1.5
B = 2.0
TOP_K = 1000


class Fixed128DiskIndex:
    def __init__(self, index_dir):
        self.path = Path(index_dir)
        self.metadata = json.loads(
            (self.path / "metadata.json").read_text()
        )

        if not self.metadata.get("completed"):
            raise RuntimeError("Index is not marked complete")

        if self.metadata.get("format") not in {
            "fixed128_disk_v1",
            "fixed_disk_v1",
        }:
            raise ValueError("Unsupported disk index format")

        self.total_docs = int(self.metadata["num_docs"])
        self.avgdl = float(self.metadata["avgdl"])
        self.latent_dim = int(self.metadata["latent_dim"])
        self.shard_count = int(self.metadata["shard_count"])
        self.terms_per_shard = int(self.metadata["terms_per_shard"])

        self.doc_lengths = np.load(
            self.path / "document_lengths.npy", mmap_mode="r"
        )
        self.term_stats = np.load(
            self.path / "term_stats.npy", mmap_mode="r"
        )

        self._shards = {}

    def _get_shard(self, shard_id):
        if shard_id not in self._shards:
            path = self.path / f"shard_{shard_id:03d}.bin"
            if path.stat().st_size == 0:
                self._shards[shard_id] = None
            else:
                self._shards[shard_id] = np.memmap(
                    path, dtype=RECORD_DTYPE, mode="r"
                )
        return self._shards[shard_id]

    def retrieve_one(self, query_terms, query_tfs, top_k=TOP_K):
        if len(query_terms) != len(query_tfs):
            raise ValueError("Query terms and weights have different lengths")

        scores = np.zeros(self.total_docs, dtype=np.float64)

        for term, tf_q in zip(query_terms, query_tfs):
            term = int(term)
            if term < 0 or term >= self.latent_dim:
                continue

            doc_freq = int(self.term_stats[term]["doc_freq"])
            if doc_freq == 0:
                continue

            idf = float(self.term_stats[term]["idf"])
            shard_id = term // self.terms_per_shard
            records = self._get_shard(shard_id)
            if records is None:
                continue

            term_values = records["term"]
            lo = int(np.searchsorted(term_values, term, side="left"))
            hi = int(np.searchsorted(term_values, term, side="right"))

            if lo == hi:
                continue

            postings = records[lo:hi]
            docids = postings["docid"].astype(np.intp, copy=False)
            tf_d = postings["weight"].astype(np.float64, copy=False)
            doc_len = self.doc_lengths[docids].astype(np.float64, copy=False)

            h_q = (float(tf_q) * (K2 + 1.0)) / (float(tf_q) + K2)
            h_d = (tf_d * (K1 + 1.0)) / (
                tf_d + K1 * (1.0 - B + B * (doc_len / self.avgdl))
            )

            np.add.at(scores, docids, idf * h_q * h_d)

        nonzero = np.flatnonzero(scores)
        if len(nonzero) == 0:
            return []

        if len(nonzero) > top_k:
            selected = np.argpartition(scores[nonzero], -top_k)[-top_k:]
            candidates = nonzero[selected]
        else:
            candidates = nonzero

        order = np.argsort(-scores[candidates], kind="stable")
        candidates = candidates[order]

        return [
            (int(docid), float(scores[docid]))
            for docid in candidates
        ]
