"""Screen P3 sparsity settings on training-derived validation candidates.

This is a fast candidate-reranking proxy, not a replacement for full CL-SR
index evaluation. Use only the training-derived validation split here; reserve
MS MARCO dev and TREC DL for final reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from p1.metrics import evaluate as rank_metrics, read_qrels
from p2.evaluate import load_split, safe_bm25
from p2.sae import load_sae


@torch.no_grad()
def encode_topk(model, embeddings, *, top_k, batch_size, device):
    """Encode a matrix once and retain only the strongest positive activations."""
    all_values, all_ids = [], []
    for start in range(0, len(embeddings), batch_size):
        batch = torch.as_tensor(
            np.array(embeddings[start:start + batch_size], copy=True),
            dtype=torch.float32,
            device=device,
        )
        activations = model.linear(batch - model.bias_pre).clamp_min(0)
        values, ids = torch.topk(activations, k=top_k, dim=1, largest=True, sorted=True)
        all_values.append(values.cpu().numpy())
        all_ids.append(ids.cpu().numpy())
    if not all_values:
        raise ValueError("Cannot encode an empty embedding array")
    return np.concatenate(all_values), np.concatenate(all_ids)


@torch.no_grad()
def encode_query_threshold(model, embeddings, *, threshold, batch_size, device):
    """Encode queries once with the frozen checkpoint threshold, independent of passage settings."""
    rows = []
    for start in range(0, len(embeddings), batch_size):
        batch = torch.as_tensor(
            np.array(embeddings[start:start + batch_size], copy=True),
            dtype=torch.float32,
            device=device,
        )
        activations = model.linear(batch - model.bias_pre)
        for vector in activations:
            active = torch.nonzero(vector > threshold, as_tuple=False).flatten()
            rows.append({int(concept): float(vector[concept]) for concept in active.tolist()})
    if not rows:
        raise ValueError("Cannot encode an empty query embedding array")
    return rows


def selected_arrays(values, ids, *, mode, settings):
    """Return cached top-activation views and a mask for one sparsity setting."""
    if mode == "fixed":
        selected_values, selected_ids = values[:, :settings["k"]], ids[:, :settings["k"]]
        keep = selected_values > 0
    else:
        count = int(settings["max_concepts"])
        selected_values, selected_ids = values[:, :count], ids[:, :count]
        keep = selected_values > settings["threshold"]
        min_count = settings["min_concepts"]
        if min_count:
            passing = keep.sum(axis=1)
            positions = np.arange(count)[None, :]
            keep |= (positions < min_count) & (passing[:, None] < min_count) & (selected_values > 0)
    return selected_values, selected_ids, keep


def sampled_stats(values, ids, *, mode, settings, latent_dim):
    selected_values, selected_ids, keep = selected_arrays(values, ids, mode=mode, settings=settings)
    lengths = keep.sum(axis=1)
    frequencies = np.bincount(selected_ids[keep], minlength=latent_dim)
    return {
        "num_docs": len(values),
        "avg_doc_length": max(float(lengths.mean()), 1e-9),
        "document_frequency": {int(i): int(n) for i, n in enumerate(frequencies) if n},
    }


def evaluate_candidate_setting(candidate_values, candidate_ids, query_codes,
                               sample_values, sample_ids, candidates, qids, qrels,
                               mode, settings, latent_dim):
    unique_candidates = np.unique(candidates)
    unique_candidates.sort()
    passage_position = {int(row): index for index, row in enumerate(unique_candidates)}
    passage_values, passage_ids, passage_keep = selected_arrays(
        candidate_values, candidate_ids, mode=mode, settings=settings,
    )
    stats = sampled_stats(sample_values, sample_ids, mode=mode, settings=settings,
                          latent_dim=latent_dim)

    run = {}
    for query_index, (qid, candidate_rows) in enumerate(zip(qids, candidates)):
        positions = [passage_position[int(row)] for row in candidate_rows]
        scores = []
        for pos in positions:
            keep = passage_keep[pos]
            document = {int(concept): float(weight)
                        for concept, weight in zip(passage_ids[pos][keep], passage_values[pos][keep])}
            scores.append(safe_bm25(query_codes[query_index], document, stats))
        order = sorted(range(len(positions)), key=lambda idx: (-scores[idx], idx))
        run[qid] = [str(int(candidate_rows[idx])) for idx in order]

    relevant_qrels = {qid: qrels[qid] for qid in qids if qid in qrels}
    metrics = rank_metrics(run, relevant_qrels)
    metrics["avg_query_concepts"] = float(np.mean([len(row) for row in query_codes]))
    metrics["avg_passage_concepts_sample"] = stats["avg_doc_length"]
    metrics["selection"] = mode
    metrics.update(settings)
    return metrics


def comma_ints(value):
    try:
        result = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("provide at least one integer")
    return result


def comma_floats(value):
    try:
        result = [float(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not result:
        raise argparse.ArgumentTypeError("provide at least one number")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True,
                        help="Contains passages.npy, validation_queries.npy, validation.cands.npy and validation_queries.ids")
    parser.add_argument("--qrels", required=True, help="Training-derived validation qrels only")
    parser.add_argument("--thresholds", type=comma_floats, required=True)
    parser.add_argument("--min-counts", type=comma_ints, default=[8, 16])
    parser.add_argument("--max-counts", type=comma_ints, default=[64, 128])
    parser.add_argument("--fixed-k", type=comma_ints, default=[32, 48, 64, 128])
    parser.add_argument("--stats-sample", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--k", type=int, default=32, help="BatchTopK metadata for released checkpoints")
    parser.add_argument("--query-threshold", type=float,
                        help="Frozen query activation threshold; defaults to checkpoint metadata")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="Optional JSON output file; otherwise print to stdout")
    args = parser.parse_args()

    if args.stats_sample < 1 or args.batch_size < 1:
        parser.error("--stats-sample and --batch-size must be positive")
    if any(x < 0 for x in args.thresholds):
        parser.error("thresholds must be non-negative")
    if any(x < 1 for x in args.fixed_k + args.max_counts) or any(x < 0 for x in args.min_counts):
        parser.error("fixed/max counts must be positive and minimum counts non-negative")
    if args.seed < 0:
        parser.error("--seed must be non-negative")

    root = Path(args.data_dir)
    passages = np.load(root / "passages.npy", mmap_mode="r")
    if passages.ndim != 2 or len(passages) == 0:
        parser.error("passages.npy must be a non-empty 2D embedding matrix")
    queries, candidates, qids = load_split(root, "validation")
    qrels = read_qrels(args.qrels)
    if len(queries) != len(qids) or len(candidates) != len(qids):
        parser.error("validation query IDs, embeddings, and candidate rows must have equal lengths")
    if set(qids) - set(qrels):
        parser.error("validation qrels are missing one or more validation query IDs")
    if len(set(qids)) != len(qids):
        parser.error("validation query IDs must be unique")
    if len(qids) == 0:
        parser.error("validation split is empty")
    if candidates.ndim != 2 or candidates.shape[1] == 0:
        parser.error("validation candidate rows must be a non-empty 2D matrix")
    if candidates.size and (candidates.min() < 0 or candidates.max() >= len(passages)):
        parser.error("candidate passage row is outside passages.npy")

    sample_rows = np.sort(np.random.default_rng(args.seed).choice(
        len(passages), min(args.stats_sample, len(passages)), replace=False,
    ))
    model, meta = load_sae(args.checkpoint, k=args.k, map_location=args.device)
    model.to(args.device).eval()
    if passages.shape[1] != model.linear.in_features or queries.shape[1] != model.linear.in_features:
        parser.error("passage/query embedding width does not match SAE input dimension")
    latent_dim = model.linear.out_features
    if any(k > latent_dim for k in args.fixed_k + args.max_counts):
        parser.error(f"fixed-k and maximum counts cannot exceed SAE latent dimension ({latent_dim})")
    if not any(min_count <= max_count for min_count in args.min_counts for max_count in args.max_counts):
        parser.error("no min-count/max-count combination is valid")

    query_threshold = args.query_threshold if args.query_threshold is not None else meta.get("threshold")
    if query_threshold is None or not np.isfinite(query_threshold) or query_threshold < 0:
        parser.error("provide a finite non-negative --query-threshold or checkpoint threshold metadata")

    top_k = max(args.fixed_k + args.max_counts)
    if top_k > latent_dim:
        parser.error(f"fixed-k and maximum counts cannot exceed SAE latent dimension ({latent_dim})")
    unique_candidates = np.unique(candidates)
    unique_candidates.sort()
    print(f"Encoding top {top_k} activations once for {len(unique_candidates):,} unique candidate passages...",
          flush=True)
    candidate_values, candidate_ids = encode_topk(
        model, passages[unique_candidates], top_k=top_k, batch_size=args.batch_size, device=args.device,
    )
    print(f"Encoding validation queries once with frozen threshold {query_threshold:.8g}...", flush=True)
    query_codes = encode_query_threshold(
        model, queries, threshold=query_threshold, batch_size=args.batch_size, device=args.device,
    )
    results = []
    print(f"Encoding top {top_k} activations for {len(sample_rows):,} corpus-statistics sample passages...",
          flush=True)
    sample_values, sample_ids = encode_topk(
        model, passages[sample_rows], top_k=top_k, batch_size=args.batch_size, device=args.device,
    )
    print("Scoring validation settings from cached activations...", flush=True)
    for k in args.fixed_k:
        results.append(evaluate_candidate_setting(
            candidate_values, candidate_ids, query_codes, sample_values, sample_ids,
            candidates, qids, qrels, "fixed", {"k": k}, latent_dim,
        ))
    for threshold in args.thresholds:
        for min_count in args.min_counts:
            for max_count in args.max_counts:
                if min_count > max_count:
                    continue
                results.append(evaluate_candidate_setting(
                    candidate_values, candidate_ids, query_codes, sample_values, sample_ids,
                    candidates, qids, qrels, "adaptive", {"threshold": threshold,
                                                           "min_concepts": min_count,
                                                           "max_concepts": max_count}, latent_dim,
                ))
    fixed_rows = [row for row in results if row["selection"] == "fixed"]
    if not fixed_rows:
        parser.error("provide at least one --fixed-k baseline")
    fixed_best = max(row["mrr_at_10"] for row in fixed_rows)
    eligible = [row for row in results if row["selection"] == "adaptive"
                and row["mrr_at_10"] >= fixed_best - 0.005]
    recommendation = min(eligible, key=lambda row: row["avg_passage_concepts_sample"]) if eligible else None
    report = {
        "scope": "validation dense-candidate reranking proxy; query encoding fixed at checkpoint threshold; confirm with full CL-SR search",
        "fixed_k_best_mrr_at_10": fixed_best,
        "query_encoding": "frozen checkpoint activation threshold",
        "query_threshold": query_threshold,
        "mrr_tolerance": 0.005,
        "recommended_adaptive": recommendation,
        "results": results,
    }
    serialized = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
