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


def kl(teacher_scores, student_scores, temperature):
    teacher = F.softmax(teacher_scores / temperature, dim=-1)
    student = F.log_softmax(student_scores / temperature, dim=-1)
    return F.kl_div(student, teacher, reduction='batchmean') * temperature ** 2


def listwise_loss(q, p, q_rep, p_rep, temperature=1.):
    """Ours: match the dense teacher's distribution over each query's own hard candidates.

    q: [B, d] original queries, p: [B, C, d] original candidates (teacher side);
    q_rep, p_rep: student representations (reconstructions or scaled sparse codes).
    """
    teacher = torch.einsum('bd,bcd->bc', q, p)
    student = torch.einsum('bd,bcd->bc', q_rep, p_rep)
    return kl(teacher, student, temperature)


def naacl_loss(q, p, q_rep, p_rep, temperature=1.):
    """Kang et al. (NAACL 2025) Eq. 2 baseline: KL over positives, normalised over the batch's positives.

    q, p: [B, d] queries and one relevant passage each; other rows act as in-batch negatives.
    """
    return kl(q @ p.T, q_rep @ p_rep.T, temperature)
