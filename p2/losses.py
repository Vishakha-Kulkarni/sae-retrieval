"""Training objectives. Every ranking loss is KL(teacher || student) on softmax score distributions."""
import torch
from torch.nn import functional as F


def reconstruction_loss(x, recon, dead_recon=None, aux_alpha=1 / 32):
    """Upstream objective: MSE plus the dead-latent auxiliary term."""
    error = x - recon
    loss = error.square()
    if dead_recon is not None:
        loss = loss + aux_alpha * (error - dead_recon).square()
    return loss.mean()


def standardize(scores, eps=1e-6):
    """Per-query z-score: keeps the ordering and relative gaps of a candidate list, drops its scale."""
    return (scores - scores.mean(-1, keepdim=True)) / (scores.std(-1, keepdim=True) + eps)


def kl(teacher_scores, student_scores, temperature, standardized=False):
    """standardized=True compares score shapes only. Without it, a teacher/student scale mismatch
    (SimLM dot products vs concept or BM25 scores) dominates the KL and is "fixed" by distorting
    activation magnitudes instead of improving the ranking."""
    if standardized:
        teacher_scores, student_scores = standardize(teacher_scores), standardize(student_scores)
    teacher = F.softmax(teacher_scores / temperature, dim=-1)
    student = F.log_softmax(student_scores / temperature, dim=-1)
    return F.kl_div(student, teacher, reduction='batchmean') * temperature ** 2


def listwise_loss(q, p, q_rep, p_rep, temperature=1., standardized=False):
    """Ours: match the dense teacher's distribution over each query's own hard candidates.

    q: [B, d] original queries, p: [B, C, d] original candidates (teacher side);
    q_rep, p_rep: student representations (reconstructions or scaled sparse codes).
    """
    teacher = torch.einsum('bd,bcd->bc', q, p)
    student = torch.einsum('bd,bcd->bc', q_rep, p_rep)
    return kl(teacher, student, temperature, standardized)


def naacl_loss(q, p, q_rep, p_rep, temperature=1., standardized=False):
    """Kang et al. (NAACL 2025) Eq. 2 baseline: KL over positives, normalised over the batch's positives.

    q, p: [B, d] queries and one relevant passage each; other rows act as in-batch negatives.
    """
    return kl(q @ p.T, q_rep @ p_rep.T, temperature, standardized)


K1, K2, B = 0.6, 1.5, 2.0  # upstream cl-sr/latent_retrieve.py


def bm25_scores(zq, zp, idf, avgdl, log_scale):
    """Differentiable CL-SR score: upstream BM25 over concept activations.

    zq: [B, V] query codes, zp: [B, C, V] candidate codes, idf: [V] fixed, avgdl: mean active concepts.
    Passage length (active-concept count) is treated as a constant, as it is a count.
    """
    dl = (zp > 0).sum(-1, keepdim=True).float()
    # 1 - b + b*dl/avgdl can approach 0 for short passages when b = 2; keep the denominator positive.
    norm = (K1 * (1 - B + B * dl / avgdl)).clamp(min=1e-2 * K1)
    hd = zp * (K1 + 1) / (zp + norm)
    hq = zq * (K2 + 1) / (zq + K2)
    return torch.einsum('bv,bcv->bc', hq * idf, hd) * log_scale.exp()
