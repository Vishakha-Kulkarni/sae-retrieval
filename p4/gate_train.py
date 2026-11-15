"""P4: train the learned concept gate.

python -m p4.gate_train --x gate_X.npz --teacher train.teacher.npy --df df_b0.npy --lam 0.1 --out gate_lam0.1

One gate per concept, between 0 and 1. A candidate's CL-SR score with gates is

    student(q, d) = sum over concepts c of  X[(q, d), c] * gate[c]

The loss makes the CL-SR ranking of each question's 100 candidates match SimLM's (teacher)
ranking, plus lambda * (how much of the index is still open), which pushes gates to close.

Outputs: <out>.gate.npy (all gate values) and <out>.drop.npy (concept ids with gate < 0.5,
pass this to pyserini_indexing --drop).
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse


def batch_matrix(X, q_start, q_end, n_cands, device):
    """Rows of X for questions q_start..q_end-1, as a torch sparse matrix on the GPU."""
    part = X[q_start * n_cands: q_end * n_cands].tocoo()
    index = torch.from_numpy(np.vstack([part.row, part.col])).long()
    values = torch.from_numpy(part.data).float()
    return torch.sparse_coo_tensor(index, values, part.shape).to(device)


def kl_loss(teacher, student):
    """How differently two score lists rank the same candidates. Both [questions, n_cands].
    Each row becomes probabilities (softmax); 0 = same ranking and same confidence."""
    return F.kl_div(F.log_softmax(student, -1), F.log_softmax(teacher, -1),
                    log_target=True, reduction='batchmean')


def fit_alpha(X, teacher, n_cands, sample=20000):
    """One scale number so that alpha * CL-SR score lines up with SimLM's score.
    Done once with all gates open (= plain CL-SR), then frozen."""
    rng = np.random.default_rng(0)
    qs = rng.choice(len(teacher), size=min(sample, len(teacher)), replace=False)
    rows = (qs[:, None] * n_cands + np.arange(n_cands)).ravel()
    clsr = torch.from_numpy(np.asarray(X[rows].sum(axis=1)).reshape(len(qs), n_cands)).float()
    t = torch.from_numpy(teacher[qs])
    alphas = np.logspace(-3, 1, 41)
    losses = [kl_loss(t, a * clsr).item() for a in alphas]
    best = int(np.argmin(losses))
    print(f'alpha search: best {alphas[best]:.4g} (KL {losses[best]:.4f}); '
          f'smallest tried {alphas[0]:.3g}, largest {alphas[-1]:.3g}', flush=True)
    return float(alphas[best])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--x', required=True, help='gate_X.npz from p4.gate_data')
    ap.add_argument('--teacher', required=True, help='train.teacher.npy')
    ap.add_argument('--df', help='optional: passages per concept (.npy). If given, the penalty counts '
                                 'postings (index size) instead of concepts')
    ap.add_argument('--lam', type=float, required=True, help='sparsity strength: bigger = more gates close')
    ap.add_argument('--out', required=True, help='output prefix')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=4096, help='questions per step')
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0, help='batch order (report 3 seeds per lambda)')
    args = ap.parse_args()
    args = ap.parse_args()

    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X = sparse.load_npz(args.x).tocsr()
    teacher = np.load(args.teacher).astype(np.float32)
    n_questions, n_cands = teacher.shape
    num_concepts = X.shape[1]
    assert X.shape[0] == n_questions * n_cands, 'gate_X rows must be questions x candidates'
    print(f'{n_questions:,} questions x {n_cands} candidates, {num_concepts:,} concepts', flush=True)

    if args.df:
        df = np.load(args.df).astype(np.float64)
        cost = torch.tensor(df / df.sum(), dtype=torch.float32, device=device)
    else:
        cost = torch.full((num_concepts,), 1 / num_concepts, device=device)

    alpha = fit_alpha(X, teacher, n_cands)

    w = torch.full((num_concepts,), 3.0, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=args.lr)

    for epoch in range(args.epochs):
        starts = np.random.permutation(np.arange(0, n_questions, args.batch))
        for step, start in enumerate(starts):
            end = min(start + args.batch, n_questions)
            Xb = batch_matrix(X, start, end, n_cands, device)                  
            tb = torch.from_numpy(teacher[start:end]).to(device)              

            gate = torch.sigmoid(w)
            student = (Xb @ gate).view(end - start, n_cands)                 
            kl = kl_loss(tb, alpha * student)
            open_share = (gate * cost).sum()                                  
            loss = kl + args.lam * open_share

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f'epoch {epoch} step {step:4d}/{len(starts)}  loss {loss.item():.4f}  '
                      f'kl {kl.item():.4f}  open {open_share.item():.3f}  '
                      f'gates<0.5: {(gate < 0.5).sum().item():,}', flush=True)

    # ---- save
    gate = torch.sigmoid(w).detach().cpu().numpy()
    drop = np.flatnonzero(gate < 0.5)
    np.save(f'{args.out}.gate.npy', gate)
    np.save(f'{args.out}.drop.npy', drop)
    msg = f'dropped {len(drop):,} of {num_concepts:,} concepts'
    if args.df:
        msg += f' = {df[drop].sum() / df.sum():.1%} of postings'
    print(f'{msg} -> {args.out}.drop.npy', flush=True)


if __name__ == '__main__':
    main()