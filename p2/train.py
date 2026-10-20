"""Ranking-aware BatchTopK SAE training: python -m p2.train --help

loss = reconstruction (+ upstream dead-latent aux) + rank_weight * ranking KL

--rank-loss none      released objective (our corrected, runnable port of upstream sae/train.py)
--rank-loss naacl     Kang et al. NAACL 2025 baseline: in-batch KL over query-positive pairs
--rank-loss listwise  ours: KL to SimLM's distribution over each query's dense top-K hard candidates
--rank-space recon    student scores from reconstructed embeddings (as NAACL)
--rank-space latent   ours: student scores from sparse concept codes, the space CL-SR searches

Expected --data-dir layout (produced by p2.encode and p2.candidates):
  passages.npy, train_queries.npy/.ids, train.cands.npy, train.teacher.npy, train.pos.npy,
  val_queries.npy/.ids, val.cands.npy
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from p1.metrics import read_qrels
from .encode import default_device
from .evaluate import candidate_metrics, encode, nmse
from .losses import listwise_loss, naacl_loss, reconstruction_loss
from .sae import SAE, load_sae


def gather(array, rows, device):
    """Random rows from a (possibly memory-mapped) array; sorted reads are much faster on disk."""
    order = np.argsort(rows)
    out = np.empty((len(rows),) + array.shape[1:], dtype=np.float32)
    out[order] = array[rows[order]]
    return torch.from_numpy(out).to(device)


def candidate_columns(rng, batch, total, keep, head):
    head = min(head, keep)
    rest = np.stack([rng.choice(np.arange(head, total), keep - head, replace=False) for _ in range(batch)]) \
        if keep > head else np.zeros((batch, 0), dtype=np.int64)
    return np.concatenate([np.tile(np.arange(head), (batch, 1)), rest], axis=1)


def ranking_step(model, args, data, rng, device, scale):
    queries, cands, pos = data['train_queries'], data['train_cands'], data['train_pos']
    if args.rank_loss == 'naacl':
        valid = data['valid_pos']
        qrows = rng.choice(valid, args.rank_batch, replace=False)
        q = gather(queries, qrows, device)
        p = gather(data['passages'], pos[qrows], device)
        q_rep, p_rep = student(model, q, args, scale), student(model, p, args, scale)
        return naacl_loss(q, p, q_rep, p_rep, args.temperature)
    qrows = rng.choice(len(queries), args.rank_batch, replace=False)
    cols = candidate_columns(rng, len(qrows), cands.shape[1], args.rank_cands, args.rank_head)
    prows = np.take_along_axis(cands[qrows], cols, 1)
    q = gather(queries, qrows, device)
    p = gather(data['passages'], prows.reshape(-1), device)
    q_rep = student(model, q, args, scale)
    p_rep = student(model, p, args, scale).view(len(qrows), cols.shape[1], -1)
    return listwise_loss(q, p.view(len(qrows), cols.shape[1], -1), q_rep, p_rep, args.temperature)


def student(model, x, args, scale):
    recon, z, _, _ = model(x, track_dead=False)
    if args.rank_space == 'recon':
        return recon
    # Concept-space dot products live on a different scale than SimLM scores; a learned scalar
    # (applied to both sides, so the score scales by exp(2 * log_scale)) absorbs the mismatch.
    return z * scale.exp()


def validate(model, threshold, data, qrels, device, args):
    rows = data['nmse_rows']
    recon, _ = encode(model, data['passages'][rows], threshold, device=device)
    result = {'nmse': nmse(data['passages'][rows], recon)}
    result.update(candidate_metrics(model, threshold, data['passages'], data['val_queries'], data['val_cands'],
                                    data['val_qids'], qrels, args.stats_sample, device))
    return result


def load_data(root, args, rng):
    root = Path(root)
    mode = None if args.in_ram else 'r'
    data = {'passages': np.load(root / 'passages.npy', mmap_mode=mode),
            'train_queries': np.load(root / 'train_queries.npy', mmap_mode='r'),
            'train_cands': np.load(root / 'train.cands.npy', mmap_mode='r'),
            'train_pos': np.load(root / 'train.pos.npy'),
            'val_queries': np.load(root / 'val_queries.npy', mmap_mode='r'),
            'val_cands': np.load(root / 'val.cands.npy'),
            'val_qids': (root / 'val_queries.ids').read_text().split()}
    train_qids = set((root / 'train_queries.ids').read_text().split())
    if train_qids & set(data['val_qids']):
        raise ValueError('Train and validation queries overlap; tuning would leak')
    if args.val_queries:
        keep = rng.choice(len(data['val_qids']), min(args.val_queries, len(data['val_qids'])), replace=False)
        keep.sort()
        data['val_queries'] = np.asarray(data['val_queries'][keep])
        data['val_cands'] = data['val_cands'][keep]
        data['val_qids'] = [data['val_qids'][i] for i in keep]
    data['valid_pos'] = np.flatnonzero(data['train_pos'] >= 0)
    data['nmse_rows'] = np.sort(rng.choice(len(data['passages']), min(args.nmse_sample, len(data['passages'])),
                                           replace=False))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--val-qrels', required=True, help='qrels for the 5,000-query validation split only')
    parser.add_argument('--output', required=True, help='Checkpoint directory')
    parser.add_argument('--init', help='Start from a checkpoint (e.g. the released SAE) instead of scratch')
    parser.add_argument('--init-threshold', type=float, help='Threshold of a released --init checkpoint')
    parser.add_argument('--hidden-mult', type=int, default=32)
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--batch', type=int, default=4096, help='Reconstruction batch (upstream 4096)')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--aux-alpha', type=float, default=1 / 32)
    parser.add_argument('--dead-threshold', type=int, default=20)
    parser.add_argument('--rank-loss', choices=['none', 'naacl', 'listwise'], default='listwise')
    parser.add_argument('--rank-space', choices=['recon', 'latent'], default='recon')
    parser.add_argument('--rank-weight', type=float, default=0.5, help='Sweep 0.1, 0.5, 1.0')
    parser.add_argument('--rank-batch', type=int, default=64, help='Queries per ranking step')
    parser.add_argument('--rank-cands', type=int, default=32, help='Candidates per query per step')
    parser.add_argument('--rank-head', type=int, default=8, help='Teacher top ranks always included')
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--eval-every', type=int, default=2000)
    parser.add_argument('--val-queries', type=int, default=1000, help='Validation subsample per eval (0 = all)')
    parser.add_argument('--nmse-sample', type=int, default=100_000)
    parser.add_argument('--stats-sample', type=int, default=200_000)
    parser.add_argument('--max-nmse-ratio', type=float, default=1.10,
                        help='Model selection: best MRR among evals with NMSE <= ratio * reference NMSE')
    parser.add_argument('--reference-nmse', type=float,
                        help='NMSE of the reconstruction-only baseline; defaults to the first evaluation')
    parser.add_argument('--in-ram', action='store_true',
                        help='Load passage embeddings into memory (8.8M x 768 float32 = 27 GB) instead of memory-mapping')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=default_device())
    args = parser.parse_args()
    if args.rank_loss == 'none':
        args.rank_weight = 0.
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    data = load_data(args.data_dir, args, rng)
    qrels = read_qrels(args.val_qrels)
    if args.init:
        model, meta = load_sae(args.init, args.k)
        model.dead_threshold = args.dead_threshold
        threshold = args.init_threshold if args.init_threshold is not None else meta.get('threshold')
        if threshold is None:
            parser.error('--init-threshold is required for released checkpoints')
    else:
        model = SAE(data['passages'].shape[1], args.hidden_mult, args.k, args.dead_threshold)
        threshold = None
    model.to(device)
    scale = torch.nn.Parameter(torch.zeros((), device=device))
    optimizer = torch.optim.AdamW(list(model.parameters()) + [scale], lr=args.lr, eps=6e-10)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'log.jsonl', 'a')
    n_p, n_q = len(data['passages']), len(data['train_queries'])
    min_acts, best, reference_nmse = [], None, args.reference_nmse
    started = time.time()

    def evaluate_and_save(step):
        nonlocal best, reference_nmse
        current = threshold if threshold is not None and not min_acts else float(np.mean(min_acts[-1000:]))
        model.eval()
        metrics = validate(model, current, data, qrels, device, args)
        model.train()
        if reference_nmse is None:
            reference_nmse = metrics['nmse']
        record = {'step': step, 'threshold': current, 'elapsed_s': round(time.time() - started), **metrics}
        log.write(json.dumps(record) + '\n')
        log.flush()
        print(json.dumps(record), flush=True)
        payload = {'state_dict': model.upstream_state_dict(), 'meta': {'k': args.k, 'threshold': current,
                   'args': vars(args), 'validation': metrics, 'step': step, 'rank_scale': scale.item()}}
        torch.save(payload, out / 'last.pt')
        eligible = metrics['nmse'] <= args.max_nmse_ratio * reference_nmse or step == 0
        score = metrics['clsr_bm25']['mrr_at_10']
        if eligible and (best is None or score > best):
            best = score
            torch.save(payload, out / 'best.pt')

    if args.init:
        evaluate_and_save(0)  # reference point: the released checkpoint before any update
    model.train()
    for step in range(1, args.steps + 1):
        # Upstream trains on passages and training queries together.
        rows = rng.integers(0, n_p + n_q, args.batch)
        is_q = rows >= n_p
        x = torch.empty((args.batch, data['passages'].shape[1]), device=device)
        if (~is_q).any():
            x[torch.from_numpy(~is_q)] = gather(data['passages'], rows[~is_q], device)
        if is_q.any():
            x[torch.from_numpy(is_q)] = gather(data['train_queries'], rows[is_q] - n_p, device)
        optimizer.zero_grad()
        recon, _, dead_recon, min_act = model(x)
        min_acts.append(min_act.item())
        loss = reconstruction_loss(x, recon, dead_recon, args.aux_alpha)
        rank = torch.zeros((), device=device)
        if args.rank_weight > 0:
            rank = ranking_step(model, args, data, rng, device, scale)
            # Ranking KL is O(1) while MSE on SimLM vectors is much smaller; normalise by x's
            # variance so rank_weight means the same thing across embedding scales.
            loss = loss + args.rank_weight * rank * x.var(0).mean().detach()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
        model.project_decoder_grad()
        optimizer.step()
        model.make_decoder_weights_norm()
        if step % 100 == 0:
            print(json.dumps({'step': step, 'loss': loss.item(), 'rank_kl': rank.item(),
                              'min_act': float(np.mean(min_acts[-100:]))}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            evaluate_and_save(step)
    log.close()


if __name__ == '__main__':
    main()
