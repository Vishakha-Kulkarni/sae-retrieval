"""Shared week-1 task: exact dense top-K candidates with SimLM scores (needed by P1 and P2).

python -m p2.candidates --passages passages.npy --queries train_queries.npy \
    --qrels qrels.train.tsv --output-prefix data/train --top-k 100
Writes <prefix>.cands.npy [Q,K] passage rows, <prefix>.teacher.npy [Q,K] dot scores,
<prefix>.pos.npy [Q] first judged-relevant passage row (-1 if none) and <prefix>.run (TREC run,
which p1.prepare groups accepts as --teacher-run).
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from .encode import default_device


def read_qrels_rows(path):
    qrels = {}
    with open(path) as source:
        for line in source:
            parts = line.split()
            if len(parts) == 4 and int(parts[3]) > 0:
                qrels.setdefault(parts[0], []).append(int(parts[2]))
    return qrels


@torch.no_grad()
def search(passages, queries, top_k=100, chunk=200_000, device=None, query_batch=4096):
    """Exact inner-product search. Each passage chunk is read from disk once and scored against
    every query batch; running top-k lists stay on the device (500K queries x 100 is ~0.6 GB)."""
    device = device or default_device()
    q = torch.from_numpy(np.ascontiguousarray(queries)).to(device)
    best_s = torch.full((len(q), top_k), -float('inf'), device=device)
    best_i = torch.zeros((len(q), top_k), dtype=torch.long, device=device)
    for start in range(0, len(passages), chunk):
        p = torch.from_numpy(np.ascontiguousarray(passages[start:start + chunk])).to(device)
        for qs in range(0, len(q), query_batch):
            s, i = torch.topk(q[qs:qs + query_batch] @ p.T, k=min(top_k, len(p)), dim=1)
            s = torch.cat([best_s[qs:qs + query_batch], s], 1)
            i = torch.cat([best_i[qs:qs + query_batch], i + start], 1)
            best_s[qs:qs + query_batch], order = torch.topk(s, k=top_k, dim=1)
            best_i[qs:qs + query_batch] = torch.gather(i, 1, order)
        print(f'{min(start + chunk, len(passages))}/{len(passages)} passages', flush=True)
    return best_i.cpu().numpy(), best_s.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--passages', required=True)
    parser.add_argument('--queries', required=True, help='.npy from p2.encode queries (with .ids sibling)')
    parser.add_argument('--qrels')
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--query-batch', type=int, default=4096)
    parser.add_argument('--device')
    args = parser.parse_args()
    passages = np.load(args.passages, mmap_mode='r')
    queries = np.load(args.queries, mmap_mode='r')
    qids = Path(args.queries).with_suffix('.ids').read_text().split()
    if args.top_k > len(passages):
        parser.error('top-k exceeds corpus size')
    cands, scores = search(passages, queries, args.top_k, device=args.device, query_batch=args.query_batch)
    scores = scores.astype(np.float32)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(f'{prefix}.cands.npy', cands)
    np.save(f'{prefix}.teacher.npy', scores)
    qrels = read_qrels_rows(args.qrels) if args.qrels else {}
    np.save(f'{prefix}.pos.npy', np.array([qrels.get(q, [-1])[0] for q in qids], dtype=np.int64))
    with open(f'{prefix}.run', 'w') as run:
        for qid, row, srow in zip(qids, cands, scores):
            for rank, (doc, score) in enumerate(zip(row, srow), 1):
                run.write(f'{qid} Q0 {doc} {rank} {score:.4f} simlm\n')


if __name__ == '__main__':
    main()
