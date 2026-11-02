"""Evaluate an SAE checkpoint and export its concepts for the shared CL-SR index.

python -m p2.evaluate metrics --checkpoint ckpt.pt --data-dir data --split val --qrels val.qrels
python -m p2.evaluate export  --checkpoint ckpt.pt --embeddings passages.npy --output passages.latent.jsonl
python -m p2.evaluate export  --checkpoint ckpt.pt --embeddings dev_queries.npy --output dev.query.json --queries

metrics reports NMSE, average active concepts, dead-latent share, and MRR@10 / NDCG@10 when the
dense top-100 candidates are re-ranked by (a) reconstructed embeddings and (b) the released CL-SR
BM25 formula over concepts. (b) is a fast proxy; final numbers come from the full CL-SR index (P4).
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from p1.data import bm25
from p1.metrics import evaluate as rank_metrics, read_qrels
from .sae import load_sae


@torch.no_grad()
def encode(model, x, threshold, batch_size=4096, device='cpu'):
    """Threshold encoding exactly as upstream extract_latent_concepts.py; returns (recon, sparse codes)."""
    recons, codes = [], []
    for start in range(0, len(x), batch_size):
        batch = torch.as_tensor(np.asarray(x[start:start + batch_size]), dtype=torch.float32, device=device)
        z = model.threshold(model.pre_activation(batch), threshold)
        recons.append(model.decode(z).cpu())
        codes.append(z.to_sparse().cpu())
    return torch.cat(recons), torch.cat(codes)


def nmse(x, recon):
    """Upstream definition: MSE(x, recon) / MSE(x, mean(x))."""
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    return ((x - recon).square().mean() / (x - x.mean(0)).square().mean()).item()


def to_dicts(codes):
    codes = codes.coalesce()
    rows, cols = codes.indices()
    values = codes.values()
    out = [dict() for _ in range(codes.shape[0])]
    for r, c, v in zip(rows.tolist(), cols.tolist(), values.tolist()):
        if v > 0:
            out[r][c] = v
    return out


def corpus_stats(passage_codes):
    frequencies = Counter()
    length = 0
    for doc in passage_codes:
        frequencies.update(doc.keys())
        length += len(doc)
    return {'num_docs': len(passage_codes), 'avg_doc_length': max(length / len(passage_codes), 1e-9),
            'document_frequency': frequencies}


def safe_bm25(query, doc, stats):
    df = stats['document_frequency']
    shared = {c: v for c, v in query.items() if c in doc and c in df}
    return bm25(shared, doc, stats) if shared else 0.


@torch.no_grad()
def candidate_metrics(model, threshold, passages, queries, cands, qids, qrels, stats_sample=200_000,
                      device='cpu', seed=0):
    """Re-rank fixed dense candidates with reconstructions and with CL-SR BM25 over concepts."""
    rows = np.unique(cands)
    p_recon, p_codes = encode(model, passages[np.sort(rows)], threshold, device=device)
    position = {int(r): i for i, r in enumerate(np.sort(rows))}
    q_recon, q_codes = encode(model, queries, threshold, device=device)
    # Corpus statistics for IDF are estimated on a uniform passage sample; the full index uses all.
    sample = np.sort(np.random.default_rng(seed).choice(len(passages), min(stats_sample, len(passages)),
                                                        replace=False))
    _, sample_codes = encode(model, passages[sample], threshold, device=device)
    stats = corpus_stats(to_dicts(sample_codes))
    p_dicts, q_dicts = to_dicts(p_codes), to_dicts(q_codes)
    runs = {'recon_dense': {}, 'clsr_bm25': {}}
    for qi, (qid, row) in enumerate(zip(qids, cands)):
        idx = [position[int(r)] for r in row]
        dense = (p_recon[idx] @ q_recon[qi]).tolist()
        sparse = [safe_bm25(q_dicts[qi], p_dicts[i], stats) for i in idx]
        for name, scores in (('recon_dense', dense), ('clsr_bm25', sparse)):
            order = sorted(range(len(row)), key=lambda j: (-scores[j], j))
            runs[name][qid] = [str(int(row[j])) for j in order]
    qrels = {q: qrels[q] for q in qids if q in qrels}
    out = {name: rank_metrics(run, qrels) for name, run in runs.items()}
    out['avg_query_concepts'] = sum(map(len, q_dicts)) / len(q_dicts)
    out['avg_passage_concepts'] = stats['avg_doc_length']
    out['dead_latent_share'] = 1 - len(stats['document_frequency']) / model.linear.weight.shape[0]
    return out


def load_split(data_dir, split):
    root = Path(data_dir)
    return (np.load(root / f'{split}_queries.npy', mmap_mode='r'),
            np.load(root / f'{split}.cands.npy'),
            (root / f'{split}_queries.ids').read_text().split())


def export(model, threshold, embeddings, output, queries=False, ids=None, device='cpu', batch_size=4096):
    """Write the upstream latent formats consumed by cl-sr/indexing.py and latent_retrieve.py."""
    ids = ids or [str(i) for i in range(len(embeddings))]
    query_out = {}
    with open(output, 'w') as target:
        for start in range(0, len(embeddings), batch_size):
            _, codes = encode(model, embeddings[start:start + batch_size], threshold, batch_size, device)
            for offset, concepts in enumerate(to_dicts(codes)):
                pairs = sorted(concepts.items(), key=lambda item: -item[1])
                record = {'ids': [c for c, _ in pairs], 'weight': [v for _, v in pairs]}
                if queries:
                    # upstream indexing.py keys postings by str(concept) and latent_retrieve.py looks
                    # query concepts up verbatim, so query IDs must be strings or nothing matches.
                    query_out[ids[start + offset]] = {'ids': [str(c) for c in record['ids']],
                                                      'weight': record['weight']}
                else:
                    target.write(json.dumps({'docid': int(ids[start + offset]), **record}) + '\n')
        if queries:
            json.dump(query_out, target)


@torch.no_grad()
def table1(model, threshold, passages, queries, qids, qrels, top_k=1000, chunk=20_000, device='cpu',
           reconstruct_queries=False):
    """Paper Table 1 protocol: exact dense search over the whole corpus with reconstructed passages
    (original queries unless reconstruct_queries), plus corpus NMSE. Passages are reconstructed chunk by chunk and never stored.
    A 20K chunk keeps the 24,576-wide activations near 2 GB; 200K needed ~18 GB per tensor (OOM on A100-40GB)."""
    keep = [i for i, q in enumerate(qids) if q in qrels]
    q = torch.as_tensor(np.asarray(queries[keep]), dtype=torch.float32, device=device)
    # Upstream Table 1 (simlm search_npy.py) searches original SimLM queries against reconstructed
    # passages; reconstruct_queries=True is a stricter variant where both sides go through the SAE.
    q_hat = model.decode(model.threshold(model.pre_activation(q), threshold)) if reconstruct_queries else q
    best_s = torch.full((len(q), top_k), -float('inf'), device=device)
    best_i = torch.zeros((len(q), top_k), dtype=torch.long, device=device)
    err = total = 0.
    # Corpus mean for the NMSE denominator, estimated on a uniform sample of passages.
    sample = np.sort(np.random.default_rng(0).choice(len(passages), min(200_000, len(passages)), replace=False))
    mean = torch.as_tensor(np.asarray(passages[sample]).mean(0), device=device)
    for start in range(0, len(passages), chunk):
        x = torch.as_tensor(np.array(passages[start:start + chunk]), dtype=torch.float32, device=device)
        x_hat = model.decode(model.threshold(model.pre_activation(x), threshold))
        err += (x - x_hat).square().sum().item()
        total += (x - mean).square().sum().item()
        s, i = torch.topk(q_hat @ x_hat.T, k=min(top_k, len(x)), dim=1)
        s, i = torch.cat([best_s, s], 1), torch.cat([best_i, i + start], 1)
        best_s, order = torch.topk(s, k=top_k, dim=1)
        best_i = torch.gather(i, 1, order)
    # Drop unfilled slots (-inf), which only exist when the corpus is smaller than top_k.
    run = {qids[j]: [str(d) for d, s in zip(row, srow) if s != -float('inf')]
           for j, row, srow in zip(keep, best_i.cpu().tolist(), best_s.cpu().tolist())}
    return {'nmse': err / total, **rank_metrics(run, {q: qrels[q] for q in run})}


def main():
    from .encode import default_device
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('metrics', 'export', 'table1'):
        command = commands.add_parser(name)
        command.add_argument('--checkpoint', required=True)
        command.add_argument('--threshold', type=float, help='Defaults to the checkpoint mean_of_min_act')
        command.add_argument('--k', type=int, default=32, help='Only for released upstream checkpoints')
        command.add_argument('--device', default=default_device())
    metrics = commands.choices['metrics']
    metrics.add_argument('--data-dir', required=True)
    metrics.add_argument('--split', default='val')
    metrics.add_argument('--qrels', required=True)
    metrics.add_argument('--nmse-sample', type=int, default=200_000)
    exporter = commands.choices['export']
    exporter.add_argument('--embeddings', required=True)
    exporter.add_argument('--output', required=True)
    exporter.add_argument('--queries', action='store_true', help='Write the query JSON dict format')
    paper = commands.choices['table1']
    paper.add_argument('--passages', required=True)
    paper.add_argument('--queries', required=True, help='e.g. dev_queries.npy (with .ids sibling)')
    paper.add_argument('--qrels', required=True, help='e.g. qrels.dev.small.tsv; only judged queries are scored')
    paper.add_argument('--reconstruct-queries', action='store_true',
                       help='Also reconstruct queries (stricter than the paper, which uses original queries)')
    args = parser.parse_args()
    model, meta = load_sae(args.checkpoint, args.k)
    model.to(args.device).eval()
    threshold = args.threshold if args.threshold is not None else meta.get('threshold')
    if threshold is None:
        parser.error('Released checkpoints need --threshold (the mean_of_min_act in their filename)')
    if args.command == 'export':
        embeddings = np.load(args.embeddings, mmap_mode='r')
        ids_path = Path(args.embeddings).with_suffix('.ids')
        ids = ids_path.read_text().split() if ids_path.exists() else None
        export(model, threshold, embeddings, args.output, args.queries, ids, args.device)
        return
    if args.command == 'table1':
        queries = np.load(args.queries, mmap_mode='r')
        qids = Path(args.queries).with_suffix('.ids').read_text().split()
        result = table1(model, threshold, np.load(args.passages, mmap_mode='r'), queries, qids,
                        read_qrels(args.qrels), device=args.device,
                        reconstruct_queries=args.reconstruct_queries)
        print(json.dumps({'checkpoint': args.checkpoint, 'threshold': threshold,
                          'queries': 'reconstructed' if args.reconstruct_queries else 'original (paper protocol)',
                          **result}, indent=1))
        return
    passages = np.load(Path(args.data_dir) / 'passages.npy', mmap_mode='r')
    queries, cands, qids = load_split(args.data_dir, args.split)
    rows = np.sort(np.random.default_rng(1).choice(len(passages), min(args.nmse_sample, len(passages)),
                                                   replace=False))
    recon, _ = encode(model, passages[rows], threshold, device=args.device)
    result = {'checkpoint': args.checkpoint, 'threshold': threshold, 'nmse': nmse(passages[rows], recon),
              **candidate_metrics(model, threshold, passages, queries, cands, qids, read_qrels(args.qrels),
                                  device=args.device)}
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
