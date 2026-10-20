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

from .adaptive_encoder import encode_adaptive_batch, encode_fixed_k_batch
from p1.metrics import evaluate as rank_metrics, read_qrels
from p2.evaluate import corpus_stats, load_split, safe_bm25
from p2.sae import load_sae


@torch.no_grad()
def encode_selection(model, embeddings, *, mode, settings, batch_size, device):
    """Return positive sparse concept dictionaries without retaining reconstructions."""
    sparse_rows = []
    for start in range(0, len(embeddings), batch_size):
        batch = torch.as_tensor(
            np.asarray(embeddings[start:start + batch_size]),
            dtype=torch.float32,
            device=device,
        )
        if mode == "adaptive":
            rows, recon = encode_adaptive_batch(
                model, batch, threshold=settings["threshold"],
                min_concepts=settings["min_concepts"],
                max_concepts=settings["max_concepts"],
                return_reconstruction=False,
            )
        else:
            rows, _ = encode_fixed_k_batch(model, batch, k=settings["k"], return_reconstruction=False)
        for _, concepts, weights in rows:
            sparse_rows.append({int(c): float(w) for c, w in zip(concepts, weights) if w > 0})
    if not sparse_rows:
        raise ValueError("Cannot encode an empty embedding array")
    return sparse_rows


@torch.no_grad()
def evaluate_candidate_setting(model, passages, queries, candidates, qids, qrels,
                               sample_rows, mode, settings, batch_size, device):
    unique_candidates = np.unique(candidates)
    unique_candidates.sort()
    passage_codes = encode_selection(
        model, passages[unique_candidates], mode=mode, settings=settings,
        batch_size=batch_size, device=device,
    )
    passage_position = {int(row): index for index, row in enumerate(unique_candidates)}
    query_codes = encode_selection(
        model, queries, mode=mode, settings=settings, batch_size=batch_size, device=device,
    )
    sample_codes = encode_selection(
        model, passages[sample_rows], mode=mode, settings=settings,
        batch_size=batch_size, device=device,
    )
    stats = corpus_stats(sample_codes)

    run = {}
    for query_index, (qid, candidate_rows) in enumerate(zip(qids, candidates)):
        positions = [passage_position[int(row)] for row in candidate_rows]
        scores = [safe_bm25(query_codes[query_index], passage_codes[pos], stats) for pos in positions]
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

    root = Path(args.data_dir)
    passages = np.load(root / "passages.npy", mmap_mode="r")
    queries, candidates, qids = load_split(root, "validation")
    qrels = read_qrels(args.qrels)
    if len(queries) != len(qids) or len(candidates) != len(qids):
        parser.error("validation query IDs, embeddings, and candidate rows must have equal lengths")
    if set(qids) - set(qrels):
        parser.error("validation qrels are missing one or more validation query IDs")
    if candidates.size and (candidates.min() < 0 or candidates.max() >= len(passages)):
        parser.error("candidate passage row is outside passages.npy")

    sample_rows = np.sort(np.random.default_rng(args.seed).choice(
        len(passages), min(args.stats_sample, len(passages)), replace=False,
    ))
    model, _ = load_sae(args.checkpoint, k=args.k, map_location=args.device)
    model.to(args.device).eval()

    results = []
    for k in args.fixed_k:
        results.append(evaluate_candidate_setting(
            model, passages, queries, candidates, qids, qrels, sample_rows,
            "fixed", {"k": k}, args.batch_size, args.device,
        ))
    for threshold in args.thresholds:
        for min_count in args.min_counts:
            for max_count in args.max_counts:
                if min_count > max_count:
                    continue
                results.append(evaluate_candidate_setting(
                    model, passages, queries, candidates, qids, qrels, sample_rows,
                    "adaptive", {"threshold": threshold, "min_concepts": min_count,
                                 "max_concepts": max_count},
                    args.batch_size, args.device,
                ))
    fixed_best = max(row["mrr_at_10"] for row in results if row["selection"] == "fixed")
    eligible = [row for row in results if row["selection"] == "adaptive"
                and row["mrr_at_10"] >= fixed_best - 0.005]
    recommendation = min(eligible, key=lambda row: row["avg_passage_concepts_sample"]) if eligible else None
    report = {
        "scope": "validation dense-candidate reranking proxy; confirm selected settings with full CL-SR search",
        "fixed_k_best_mrr_at_10": fixed_best,
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
