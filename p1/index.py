"""Build a bounded-memory concept posting index shared by BM25 and P1."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .prepare import latent


def records(path, cap, digest=None):
    with open(path, 'rb') as source:
        for row, line in enumerate(source):
            if digest is not None:
                digest.update(line)
            record = json.loads(line)
            if str(record['docid']) != str(row):
                raise ValueError('Passage IDs must equal sequential embedding rows')
            yield row, latent(record, cap)


def build(path, output, cap=24, batch_docs=50000):
    if min(cap, batch_docs) < 1 or cap > 65535:
        raise ValueError('Require positive cap/batch size and cap <= 65535')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    df, lengths = Counter(), []
    digest = hashlib.sha256()
    for row, concepts in records(path, cap, digest):
        df.update(concepts.keys())
        lengths.append(len(concepts))
        if (row + 1) % 500000 == 0:
            print(f'Index counts: {row + 1:,} passages', flush=True)
    if not df or len(lengths) >= 2 ** 32:
        raise ValueError('Empty index or passage IDs exceed uint32')
    width = max(df) + 1
    counts = np.array([df.get(c, 0) for c in range(width)], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    np.save(root / 'offsets.npy', offsets)
    np.save(root / 'lengths.npy', np.asarray(lengths, dtype=np.uint16))
    docs = np.lib.format.open_memmap(root / 'docs.npy', mode='w+', dtype=np.uint32, shape=(int(offsets[-1]),))
    weights = np.lib.format.open_memmap(root / 'weights.npy', mode='w+', dtype=np.float32, shape=docs.shape)
    pointers = offsets[:-1].copy()
    cols, vals = [], []
    first_row = 0

    def flush(end):
        nonlocal first_row, cols, vals
        if cols:
            columns = np.asarray(cols, dtype=np.int32)
            values = np.asarray(vals, dtype=np.float32)
            rows = np.repeat(np.arange(first_row, end, dtype=np.uint32), lengths[first_row:end])
            order = np.argsort(columns, kind='stable')
            columns, values, rows = columns[order], values[order], rows[order]
            concepts, starts, sizes = np.unique(columns, return_index=True, return_counts=True)
            for concept, start, size in zip(concepts, starts, sizes):
                position = pointers[concept]
                docs[position:position + size] = rows[start:start + size]
                weights[position:position + size] = values[start:start + size]
                pointers[concept] += size
        first_row, cols, vals = end, [], []

    count = 0
    second_digest = hashlib.sha256()
    for row, concepts in records(path, cap, second_digest):
        if row >= len(lengths) or len(concepts) != lengths[row]:
            raise ValueError('Passage input changed between index passes')
        cols.extend(concepts.keys())
        vals.extend(concepts.values())
        count += 1
        if count % batch_docs == 0:
            flush(count)
        if count % 500000 == 0:
            print(f'Index postings: {count:,} passages', flush=True)
    flush(count)
    if count != len(lengths) or not np.array_equal(pointers, offsets[1:]) or digest.digest() != second_digest.digest():
        raise ValueError('Index counts and postings disagree')
    docs.flush()
    weights.flush()
    stats = {'num_docs': count, 'avg_doc_length': sum(lengths) / count,
             'length_definition': 'concept_count', 'max_concepts': cap,
             'document_frequency': dict(df)}
    (root / 'stats.json').write_text(json.dumps(stats) + '\n')
    metadata = {'format_version': 1, 'num_docs': count, 'width': width,
                'postings': int(offsets[-1]), 'max_concepts': cap,
                'source_sha256': digest.hexdigest(),
                'index_bytes': sum(p.stat().st_size for p in root.glob('*.npy'))}
    # Written last: an interrupted build cannot be mistaken for a complete index.
    (root / 'index.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


class PostingIndex:
    def __init__(self, root):
        from .data import load_stats
        root = Path(root)
        self.metadata = json.loads((root / 'index.json').read_text())
        if self.metadata['format_version'] != 1:
            raise ValueError('Unsupported posting index version')
        self.metadata['posting_array_bytes'] = sum(p.stat().st_size for p in root.glob('*.npy'))
        self.metadata['index_bytes'] = sum(p.stat().st_size for p in root.iterdir() if p.is_file())
        self.stats = load_stats(root / 'stats.json')
        self.offsets = np.load(root / 'offsets.npy', mmap_mode='r')
        self.docs = np.load(root / 'docs.npy', mmap_mode='r')
        self.weights = np.load(root / 'weights.npy', mmap_mode='r')
        self.lengths = np.load(root / 'lengths.npy', mmap_mode='r')

    def postings(self, concept):
        if concept < 0 or concept + 1 >= len(self.offsets):
            return self.docs[:0], self.weights[:0]
        start, end = self.offsets[concept:concept + 2]
        return self.docs[start:end], self.weights[start:end]

    def close(self):
        for array in (self.offsets, self.docs, self.weights, self.lengths):
            mapping = getattr(array, '_mmap', None)
            if mapping is not None and not mapping.closed:
                mapping.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--passages', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-concepts', type=int, default=24)
    parser.add_argument('--batch-docs', type=int, default=50000)
    args = parser.parse_args()
    print(json.dumps(build(args.passages, args.output_dir, args.max_concepts, args.batch_docs), indent=2))


if __name__ == '__main__':
    main()
