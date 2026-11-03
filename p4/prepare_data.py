"""P4: build the training data for the learned concept gate.

python -m p4.gate_data --latents data/passages.latent.jsonl --queries data/train_b0.query.json \
    --ids data/train_queries.ids --cands data/train.cands.npy --out p4/test/gate_X.npz

For every training question and each of its candidate passages, it computes how many CL-SR points
each shared concept adds:

    points(q, d, c) = idf(c) * question_part(q, c) * passage_part(d, c)

using the tuned index settings (k1, b, k2, top 24 concepts per passage). Concepts with idf <= 0 are
left out, as the search does. The number of passages containing each concept (for idf) is
counted here, after the top-24 trim, exactly as the index counts it.

Output: one sparse matrix, shape [questions * candidates, concepts]. Row k is question k // 100,
candidate k % 100, in the same order as train.cands.npy. Summing a row gives that passage's normal
CL-SR score. The targets are train.teacher.npy, unchanged.
"""
import argparse
from pathlib import Path

import numpy as np
import orjson
from scipy import sparse


def passage_parts(path, num_concepts, max_latents, k1, b):
    """Every passage's top concepts and their passage part, as one sparse matrix [passages, concepts]."""
    indptr, indices, weights = [0], [], []
    with open(path, 'rb') as f:
        for line in f:
            r = orjson.loads(line)
            ids = np.asarray(r['ids'], dtype=np.int64)
            z = np.asarray(r['weight'], dtype=np.float64)
            keep = np.argsort(-z, kind='stable')[:max_latents]     
            indices.append(ids[keep])
            weights.append(z[keep])
            indptr.append(indptr[-1] + len(keep))
    indptr = np.asarray(indptr)
    lengths = np.diff(indptr)
    avgdl = lengths.mean()
    z = np.concatenate(weights)
    L = np.repeat(lengths, lengths)                                
    part = z * (1 + k1) / (z + k1 * (1 - b + b * L / avgdl))
    D = sparse.csr_matrix((part.astype(np.float32), np.concatenate(indices), indptr),
                          shape=(len(lengths), num_concepts))
    return D, avgdl


def question_parts(path, qids, idf, num_concepts, k2):
    """Every question's concepts, as idf * question part, one row per question in `qids` order."""
    data = orjson.loads(Path(path).read_bytes())
    rows, cols, vals = [], [], []
    for row, qid in enumerate(qids):
        r = data[qid]
        ids = np.asarray(r['ids'], dtype=np.int64)
        z = np.asarray(r['weight'], dtype=np.float64)
        rows.append(np.full(len(ids), row))
        cols.append(ids)
        vals.append(idf[ids] * z * (1 + k2) / (z + k2))
    Q = sparse.csr_matrix((np.concatenate(vals).astype(np.float32), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(qids), num_concepts))
    Q.data[Q.data < 0] = 0                                         
    Q.eliminate_zeros()
    return Q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--latents', required=True, help='passage concepts (.jsonl)')
    ap.add_argument('--queries', required=True, help='training question concepts (.json)')
    ap.add_argument('--ids', required=True, help='question ids, one per line, same order as --cands')
    ap.add_argument('--cands', required=True, help='candidate passage ids [questions, 100] (.npy)')
    ap.add_argument('--out', required=True, help='output file (.npz)')
    ap.add_argument('--max-latents', type=int, default=24)
    ap.add_argument('--k1', type=float, default=0.6)
    ap.add_argument('--b', type=float, default=1.5)
    ap.add_argument('--k2', type=float, default=2.0)
    ap.add_argument('--num-concepts', type=int, default=24576, help='SAE width (32 x 768 for k=32 B0)')
    args = ap.parse_args()

    qids = Path(args.ids).read_text().split()
    cands = np.load(args.cands)
    num_concepts = args.num_concepts

    print('passage parts...', flush=True)
    D, avgdl = passage_parts(args.latents, num_concepts, args.max_latents, args.k1, args.b)
    num_docs = D.shape[0]
    df = np.bincount(D.indices, minlength=num_concepts).astype(np.float64)   # passages per concept
    idf = np.log((num_docs - df + 0.5) / (df + 0.5))               # released code's idf

    print('question parts...', flush=True)
    Q = question_parts(args.queries, qids, idf, num_concepts, args.k2)

    print('pairs...', flush=True)
    n_cands, pieces = cands.shape[1], []
    for start in range(0, len(qids), 20000):
        rows = np.arange(start, min(start + 20000, len(qids)))
        piece = Q[np.repeat(rows, n_cands)].multiply(D[cands[rows].ravel()]).tocsr()
        piece.eliminate_zeros()
        pieces.append(piece)
    X = sparse.vstack(pieces, format='csr')

    sparse.save_npz(args.out, X)
    print(f'saved {args.out}: {X.shape[0]:,} pairs, {X.nnz:,} values, '
          f'{X.nnz / X.shape[0]:.2f} shared concepts per pair, avg passage length {avgdl:.2f}')


if __name__ == '__main__':
    main()
    
    