"""Full-corpus retrieval from the same concept postings with BM25 or learned impacts."""
import argparse
import json
import math
import random
from pathlib import Path
import time

import numpy as np
import torch

from .data import idf, sparse
from .index import PostingIndex
from .metrics import evaluate, read_qrels
from .model import ImpactScorer
from .train import file_hash
from .prepare import latent


class Retriever:
    def __init__(self, index, checkpoint=None, device='cpu', k1=.6, b=2., k2=1.5, chunk=262144):
        if chunk < 1 or k1 <= 0 or k2 <= 0 or b < 0:
            raise ValueError('Invalid retrieval settings')
        self.index, self.device, self.chunk = index, torch.device(device), chunk
        if self.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
        self.k1, self.b, self.k2, self.model = k1, b, k2, None
        self.lengths = torch.tensor(np.asarray(index.lengths, dtype=np.float64), device=self.device)
        self.relative_length = self.lengths / index.stats['avg_doc_length']
        if checkpoint:
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if payload.get('format_version') != 2 or payload['stats'] != index.stats:
                raise ValueError('Checkpoint feature version or corpus statistics differ from index')
            self.model = ImpactScorer(payload['hidden']).to(self.device).eval()
            self.model.load_state_dict(payload['state_dict'])
        self.scores = torch.zeros(len(index.lengths), dtype=torch.float64, device=self.device)
        self.touched = torch.zeros(len(index.lengths), dtype=torch.bool, device=self.device)

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def search(self, query, top_k=1000):
        if top_k < 1:
            raise ValueError('top-k must be positive')
        query = sparse(query)
        self.sync()
        start = time.perf_counter()
        self.scores.zero_()
        self.touched.zero_()
        postings_touched = 0
        for concept in sorted(query):
            docids, weights = self.index.postings(concept)
            postings_touched += len(docids)
            if not len(docids):
                continue
            rarity = idf(concept, self.index.stats)
            activation = query[concept]
            for position in range(0, len(docids), self.chunk):
                ids = torch.tensor(np.asarray(docids[position:position + self.chunk], dtype=np.int64), device=self.device)
                values = torch.tensor(np.asarray(weights[position:position + self.chunk]), device=self.device)
                if self.model is None:
                    values = values.double()
                    denominator = values + self.k1 * (1 - self.b + self.b * self.relative_length[ids])
                    if torch.any(denominator == 0):
                        raise ValueError('Released BM25 denominator is zero')
                    impacts = rarity * (activation * (self.k2 + 1) / (activation + self.k2)) * values * (self.k1 + 1) / denominator
                else:
                    rows = torch.stack((torch.full_like(values, math.log1p(activation)),
                                        torch.log1p(values), torch.full_like(values, rarity),
                                        torch.log1p(self.relative_length[ids]).float()), dim=1)
                    impacts = self.model.network(rows).flatten().double()
                self.scores.index_add_(0, ids, impacts)
                self.touched[ids] = True
        self.scores.masked_fill_(~self.touched, -float('inf'))
        count = min(top_k, len(self.scores))
        values, ids = torch.topk(self.scores, count)
        finite = torch.isfinite(values)
        values, ids = values[finite], ids[finite]
        # Resolve cutoff ties by numeric passage ID, including ties omitted by topk.
        if len(values):
            cutoff = values[-1]
            above = self.scores > cutoff
            selected = above.nonzero().flatten()
            ties = (self.scores == cutoff).nonzero().flatten()
            ids = torch.cat((selected, ties[:len(values) - len(selected)]))
            values = self.scores[ids]
        docids, values = ids.cpu().numpy(), values.cpu().numpy()
        order = np.lexsort((docids, -values))
        ranking = [(str(int(docids[i])), float(values[i])) for i in order]
        self.sync()
        return ranking, {'latency_ms': (time.perf_counter() - start) * 1000,
                         'postings_touched': postings_touched}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', required=True)
    p.add_argument('--queries', required=True, help='Frozen query activation JSON')
    p.add_argument('--qrels', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--device', default='cpu')
    p.add_argument('--k1', type=float, default=.6)
    p.add_argument('--b', type=float, default=2.)
    p.add_argument('--k2', type=float, default=1.5)
    p.add_argument('--top-k', type=int, default=1000)
    p.add_argument('--chunk', type=int, default=262144)
    p.add_argument('--output-prefix', required=True)
    p.add_argument('--gain', choices=['linear', 'exponential'], default='linear')
    p.add_argument('--limit', type=int, help='Sample N query IDs for validation screening only')
    p.add_argument('--sample-seed', type=int, default=42)
    args = p.parse_args()
    torch.set_num_threads(4)
    index = PostingIndex(args.index)
    retriever = Retriever(index, args.checkpoint, args.device, args.k1, args.b, args.k2, args.chunk)
    queries = json.loads(Path(args.queries).read_text())
    qrels = read_qrels(args.qrels)
    if not set(qrels) <= queries.keys():
        raise ValueError('Query activations missing evaluation query IDs')
    qids = sorted(qrels)
    if args.limit is not None:
        if args.limit < 1:
            p.error('limit must be positive')
        qids = sorted(random.Random(args.sample_seed).sample(qids, min(args.limit, len(qids))))
        qrels = {q: qrels[q] for q in qids}
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    run, latencies, touches = {}, [], []
    # Exclude one warmup retrieval from latency summaries, for both scorers.
    retriever.search(latent(queries[qids[0]]), args.top_k)
    with open(str(prefix) + '.run', 'x') as target:
        for i, qid in enumerate(qids):
            q = queries[qid]
            ranking, timing = retriever.search(latent(q), args.top_k)
            run[qid] = [docid for docid, _ in ranking]
            latencies.append(timing['latency_ms'])
            touches.append(timing['postings_touched'])
            for rank, (docid, score) in enumerate(ranking, 1):
                target.write(f'{qid} Q0 {docid} {rank} {score:.12g} p1full\n')
            if (i + 1) % 100 == 0:
                print(f'Retrieved {i + 1}/{len(qids)} queries', flush=True)
    result = {'method': 'learned' if args.checkpoint else 'bm25', 'device': str(retriever.device),
              'scope': 'Full-corpus concept retrieval', 'ndcg_gain': args.gain,
              'gpu_name': torch.cuda.get_device_name(retriever.device) if retriever.device.type == 'cuda' else None,
              'sample_limit': args.limit, 'sample_seed': args.sample_seed if args.limit else None,
              'latency_scope': 'Warm retrieval including posting transfers and ranking; excludes text/SAE encoding and index load',
              'index_bytes': index.metadata['index_bytes'],
              'checkpoint_bytes': Path(args.checkpoint).stat().st_size if args.checkpoint else 0,
              'checkpoint_sha256': file_hash(args.checkpoint) if args.checkpoint else None,
              'query_latents_sha256': file_hash(args.queries), 'qrels_sha256': file_hash(args.qrels),
              'index_source_sha256': index.metadata['source_sha256'],
              'tie_break': 'ascending numeric passage ID',
              'precision': 'float32 neural impacts, float64 accumulation; TF32 disabled',
              'peak_cuda_bytes': torch.cuda.max_memory_allocated(retriever.device) if retriever.device.type == 'cuda' else 0,
              'avg_concepts_per_passage': index.stats['avg_doc_length'],
              'mean_latency_ms': float(np.mean(latencies)), 'p95_latency_ms': float(np.percentile(latencies, 95)),
              'mean_postings_touched': float(np.mean(touches)),
              'k1': args.k1, 'b': args.b, 'k2': args.k2, **evaluate(run, qrels, args.gain)}
    Path(str(prefix) + '.metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
