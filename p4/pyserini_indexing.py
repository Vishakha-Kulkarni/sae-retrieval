"""P4: build the CL-SR concept index with Pyserini (Lucene impact index), the toolkit the paper names.

python -m p4.pyserini_indexing --latents passages.latent.jsonl --output pyserini/ --max-latents 24
python -m p4.pyserini_indexing --latents passages.latent.jsonl --output pruned/ --drop prune/df_top1pct.npy

Steps: (1) one pass over the latents for corpus statistics (document frequency, lengths);
(2) write each passage's top --max-latents concepts as a JsonVectorCollection document whose
weights are the passage-side BM25 factor; (3) run Pyserini's impact indexer; (4) report sizes.

CL-SR scores s(q, d) = sum over shared concepts of idf * h_q * h_d (released cl-sr/latent_retrieve.py).
h_d depends only on the passage, so it is stored in the index; idf * h_q is put on the query side
by p4 search. Lucene stores integer impacts, so both sides are scaled by --scale and rounded.
--length count matches the released code; --length l1 follows the paper (sum of activations).
--drop removes the listed concepts from every passage (P4 pruning). It is applied after the
--max-latents truncation, and passage lengths (and their average) are taken from before the drop,
so a pruned index is exactly the baseline index minus those concepts' postings: every remaining
posting keeps its score (--renormalize recomputes lengths instead). Search needs no change: pruned concepts have
df = 0 in the new df.npy, so p4.pyserini_search skips them in queries too.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

try:
    import orjson
    loads = orjson.loads
except ImportError:
    loads = json.loads

K1, B = float(os.environ.get("K1", 0.6)), float(os.environ.get("B", 2.0))  # released repo: cl-sr/latent_retrieve.py (k1, k2, b = 0.6, 1.5, 2.0)


def top_concepts(record, max_latents, drop=None):
    """Top concepts after --drop, plus the weights before --drop (used for the passage length, so
    pruning removes postings without changing the scores of the concepts that remain)."""
    ids = np.asarray(record['ids'], dtype=np.int64)
    weights = np.asarray(record['weight'], dtype=np.float64)
    if max_latents is not None and len(ids) > max_latents:
        keep = np.argsort(-weights, kind='stable')[:max_latents]
        ids, weights = ids[keep], weights[keep]
    before_drop = weights
    if drop is not None and len(ids):
        keep = ~drop[np.minimum(ids, len(drop) - 1)] | (ids >= len(drop))
        ids, weights = ids[keep], weights[keep]
    return ids, weights, before_drop


def load_drop(path):
    """Concept ids to prune (.npy array or text file, one id per line) -> boolean mask."""
    ids = np.load(path) if str(path).endswith('.npy') else np.loadtxt(path, dtype=np.int64, ndmin=1)
    ids = np.asarray(ids, dtype=np.int64)
    mask = np.zeros(max(int(ids.max()) + 1 if len(ids) else 1, 1 << 16), dtype=bool)
    mask[ids] = True
    return mask


def doc_length(weights, mode):
    return float(len(weights)) if mode == 'count' else float(weights.sum())


def corpus_stats(path, max_latents, length_mode, num_concepts=None, drop=None, renormalize=False):
    """Pass 1: document frequency per concept and average passage length (after truncation)."""
    df = np.zeros(num_concepts or 1 << 16, dtype=np.int64)
    total_length, num_docs = 0.0, 0
    with open(path, 'rb') as source:
        for row, line in enumerate(source):
            record = loads(line)
            if int(record['docid']) != row:
                raise ValueError(f'docid {record["docid"]} on line {row}: passage IDs must equal row positions')
            ids, weights, before_drop = top_concepts(record, max_latents, drop)
            if len(ids) and ids.max() >= len(df):
                df = np.resize(df, max(int(ids.max()) + 1, 2 * len(df)))
            np.add.at(df, ids, 1)
            total_length += doc_length(weights if renormalize else before_drop, length_mode)
            num_docs += 1
            if num_docs % 1_000_000 == 0:
                print(f'stats: {num_docs:,} passages', flush=True)
    used = int(np.flatnonzero(df).max()) + 1 if df.any() else 0
    return {'df': df[:max(used, num_concepts or 0)], 'num_docs': num_docs, 'avgdl': total_length / num_docs}


def passage_factor(weights, length, avgdl, k1=K1, b=B):
    """h_d from released latent_retrieve.bm25_score."""
    return weights * (k1 + 1.0) / (weights + k1 * (1 - b + b * length / avgdl))


def write_collection(path, out_dir, stats, args, docs_per_file=1_000_000, drop=None):
    """Pass 2: JsonVectorCollection with integer passage-side impacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target, dropped = None, 0
    with open(path, 'rb') as source:
        for row, line in enumerate(source):
            if row % docs_per_file == 0:
                if target:
                    target.close()
                target = open(out_dir / f'part{row // docs_per_file:03d}.jsonl', 'w')
            ids, weights, before_drop = top_concepts(loads(line), args.max_latents, drop)
            if args.weighting == 'bm25':
                length = doc_length(weights if args.renormalize else before_drop, args.length)
                values = passage_factor(weights, length, stats['avgdl'])
            else:
                values = weights
            impacts = np.rint(values * args.scale).astype(np.int64)
            dropped += int((impacts < 1).sum())
            vector = {str(c): int(max(v, 1)) for c, v in zip(ids, impacts)}
            target.write(json.dumps({'id': str(row), 'contents': '', 'vector': vector}) + '\n')
            if (row + 1) % 1_000_000 == 0:
                print(f'collection: {row + 1:,} passages', flush=True)
    if target:
        target.close()
    return dropped


def run_indexer(collection, index, threads):
    command = [sys.executable, '-m', 'pyserini.index.lucene', '--collection', 'JsonVectorCollection',
               '--input', str(collection), '--index', str(index), '--generator', 'DefaultLuceneDocumentGenerator',
               '--threads', str(threads), '--impact', '--pretokenized']
    print(' '.join(command), flush=True)
    subprocess.run(command, check=True)


def directory_bytes(path):
    return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--latents', required=True, help='passages.latent.jsonl (upstream format)')
    parser.add_argument('--output', required=True, help='Directory for collection/, index/, meta.json, df.npy')
    parser.add_argument('--max-latents', type=int, default=24, help='Efficient 24, Max 65 (paper Sec. 5.1)')
    parser.add_argument('--weighting', choices=['bm25', 'raw'], default='bm25',
                        help='bm25: store h_d (reproduces CL-SR scoring); raw: store activations')
    parser.add_argument('--length', choices=['count', 'l1'], default='count',
                        help='Passage length for BM25: released code (count) or paper (l1)')
    parser.add_argument('--scale', type=float, default=100., help='Quantization: impact = round(scale * value)')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--keep-collection', action='store_true', help='Keep the intermediate JSON files')
    parser.add_argument('--drop', help='Concept ids to prune (.npy or one id per line), e.g. from p4.prune_df')
    parser.add_argument('--renormalize', action='store_true',
                        help='Recompute passage lengths after --drop (changes the scores of the concepts that '
                             'remain). Default: lengths from before the drop, so pruning only removes postings')
    args = parser.parse_args()
    started = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    drop = load_drop(args.drop) if args.drop else None
    stats = corpus_stats(args.latents, args.max_latents, args.length, drop=drop, renormalize=args.renormalize)
    np.save(out / 'df.npy', stats['df'])
    dropped = write_collection(args.latents, out / 'collection', stats, args, drop=drop)
    run_indexer(out / 'collection', out / 'index', args.threads)
    if not args.keep_collection:
        shutil.rmtree(out / 'collection')
    df = stats['df']
    meta = {'num_docs': stats['num_docs'], 'avgdl': stats['avgdl'], 'max_latents': args.max_latents,
            'weighting': args.weighting, 'length': args.length, 'scale': args.scale, 'k1': K1, 'b': B,
            'postings': int(df.sum()), 'used_concepts': int((df > 0).sum()),
            'impacts_rounded_up_to_1': dropped, 'drop_file': args.drop,
            'pruned_concepts': int(drop.sum()) if drop is not None else 0,
            'renormalize': args.renormalize, 'index_bytes': directory_bytes(out / 'index'),
            'index_gb': round(directory_bytes(out / 'index') / 1e9, 3), 'build_s': round(time.time() - started)}
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()