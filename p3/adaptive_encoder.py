"""Variable sparsity passage encoding for the released BatchTopK SAE.

The returned rows match the upstream ``encode_batchtopk`` / CL-SR contract:
``[batch_index, concept_ids, activation_weights]``. Selection is performed
after the SAE's learned encoder, so the SAE checkpoint and decoder stay fixed.
"""

from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def encode_adaptive_batch(
    model: Any,
    embeddings: torch.Tensor,
    *,
    threshold: float,
    min_concepts: int,
    max_concepts: int,
    return_reconstruction: bool = True,
) -> tuple[list[list[Any]], torch.Tensor | None]:
    """Encode embeddings with threshold selection and per-row count bounds.

    For each row, retain positive SAE activations strictly above ``threshold``
    up to ``max_concepts``. If fewer than ``min_concepts`` survive, fill from
    that row's strongest remaining positive activations. This implements a global
    validation-tuned activation threshold with per-text minimum/maximum counts.

    Args:
        model: Released ``SAE_bias_pre`` model (must expose ``linear``,
            ``linear2`` and ``bias_pre``).
        embeddings: Dense input tensor with shape ``[batch, input_dim]``.
        threshold: Scalar activation threshold selected on validation data.
        min_concepts: Minimum number of selected concepts per row.
        max_concepts: Maximum number of selected concepts per row.
        return_reconstruction: Also decode the selected sparse activations.

    Returns:
        ``(rows, reconstruction)``. Each row is
        ``[row_index, concept_ids, activation_weights]`` and can be written to
        the JSONL format consumed by upstream ``cl-sr/indexing.py``. The
        reconstruction is ``None`` when ``return_reconstruction`` is false.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [batch, input_dim]")
    if not torch.isfinite(torch.tensor(float(threshold))).item() or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    if min_concepts < 0 or max_concepts < 1 or min_concepts > max_concepts:
        raise ValueError("require 0 <= min_concepts <= max_concepts and max_concepts >= 1")

    activations = model.linear(embeddings - model.bias_pre)
    latent_dim = activations.shape[1]
    if max_concepts > latent_dim:
        raise ValueError(f"max_concepts ({max_concepts}) exceeds latent dimension ({latent_dim})")

    values, ids = torch.topk(activations, k=max_concepts, dim=1, largest=True, sorted=True)
    keep = values > threshold
    counts = keep.sum(dim=1)
    if min_concepts:
        positions = torch.arange(max_concepts, device=values.device).unsqueeze(0)
        fill_minimum = ((positions < min_concepts)
                        & (counts.unsqueeze(1) < min_concepts)
                        & (values > 0))
        keep = keep | fill_minimum

    sparse = torch.where(keep, values, torch.zeros_like(values))
    rows: list[list[Any]] = []
    for row_idx in range(embeddings.shape[0]):
        row_keep = keep[row_idx]
        rows.append([
            row_idx,
            ids[row_idx][row_keep].tolist(),
            values[row_idx][row_keep].tolist(),
        ])

    reconstruction = None
    if return_reconstruction:
        dense_sparse = torch.zeros_like(activations)
        dense_sparse.scatter_(1, ids, sparse)
        reconstruction = model.linear2(dense_sparse) + model.bias_pre
    return rows, reconstruction


@torch.no_grad()
def encode_fixed_k_batch(
    model: Any,
    embeddings: torch.Tensor,
    *,
    k: int,
    return_reconstruction: bool = True,
) -> tuple[list[list[Any]], torch.Tensor | None]:
    """Fixed top-k comparison encoder using the same output contract."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [batch, input_dim]")
    activations = model.linear(embeddings - model.bias_pre)
    if k < 1 or k > activations.shape[1]:
        raise ValueError(f"k must be between 1 and latent dimension ({activations.shape[1]})")
    values, ids = torch.topk(activations, k=k, dim=1, largest=True, sorted=True)
    values = values.clamp_min(0)
    keep = values > 0
    rows = [
        [row_idx, ids[row_idx][keep[row_idx]].tolist(), values[row_idx][keep[row_idx]].tolist()]
        for row_idx in range(embeddings.shape[0])
    ]
    reconstruction = None
    if return_reconstruction:
        sparse = torch.zeros_like(activations)
        sparse.scatter_(1, ids, values)
        reconstruction = model.linear2(sparse) + model.bias_pre
    return rows, reconstruction
