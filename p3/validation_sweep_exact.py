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


def load_exact_stats(root, mode, settings, latent_dim):
    """Load full-corpus BM25 statistics for one sparsity configuration."""

    index_root = root / "p3_full_eval"

    if mode == "fixed":
        k = settings["k"]
        index_dir = index_root / f"fixed{k}.diskindex"

        metadata = json.loads(
            (index_dir / "metadata.json").read_text()
        )
        term_stats = np.load(
            index_dir / "term_stats.npy",
            mmap_mode="r",
        )

        frequencies = term_stats["doc_freq"]
        avgdl = float(metadata["avgdl"])
        num_docs = int(metadata["num_docs"])

    else:
        threshold = settings["threshold"]
        min_count = settings["min_concepts"]
        max_count = settings["max_concepts"]

        stats_dir = (
            index_root
            / "adaptive_exact_stats"
            / f"threshold_{threshold:.5f}"
        )

        metadata = json.loads(
            (stats_dir / "metadata.json").read_text()
        )
        frequencies = np.load(
            stats_dir / "document_frequency.npy",
            mmap_mode="r",
        )

        avgdl = float(metadata["avg_doc_length"])
        num_docs = int(metadata["num_docs"])

        if (
            metadata["min_concepts"] != min_count
            or metadata["max_concepts"] != max_count
        ):
            raise ValueError(
                f"Adaptive statistics settings do not match: {stats_dir}"
            )

    if len(frequencies) != latent_dim:
        raise ValueError("Document-frequency array has incorrect dimension")

    return {
        "num_docs": num_docs,
        "avg_doc_length": avgdl,
        "document_frequency": {
            int(i): int(n)
            for i, n in enumerate(frequencies)
            if n
        },
    }


def evaluate_candidate_setting(
            candidate_values, candidate_ids, query_codes,
            candidates, qids, qrels,
            mode, settings, latent_dim, stats,
        ):
    unique_candidates = np.unique(candidates)
    unique_candidates.sort()
    passage_position = {int(row): index for index, row in enumerate(unique_candidates)}
    passage_values, passage_ids, passage_keep = selected_arrays(
        candidate_values, candidate_ids, mode=mode, settings=settings,
    )

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
    metrics["avg_passage_concepts_full_corpus"] = stats["avg_doc_length"]
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

    if top_k > 128:
        parser.error("The existing top128 cache supports at most 128 concepts")

    unique_candidates = np.unique(candidates)
    unique_candidates.sort()

    cache_dir = root / "p3_full_eval" / "top128_cache"

    print("Loading existing top128 activation cache...", flush=True)

    cache_values = np.load(cache_dir / "values.npy", mmap_mode="r")
    cache_ids = np.load(cache_dir / "ids.npy", mmap_mode="r")

    if cache_values.shape[0] != len(passages):
        parser.error("Activation cache passage count does not match passages.npy")

    if cache_values.shape[1] < top_k:
        parser.error("Activation cache does not contain enough concepts")

    print(
        f"Loading activations for {len(unique_candidates):,} unique candidates...",
        flush=True,
    )

    candidate_values = np.asarray(
        cache_values[unique_candidates, :top_k]
    )
    candidate_ids = np.asarray(
        cache_ids[unique_candidates, :top_k]
    )

    print(
        f"Encoding validation queries with frozen threshold "
        f"{query_threshold:.5f}...",
        flush=True,
    )

    query_codes = encode_query_threshold(
        model, queries,
        threshold=query_threshold,
        batch_size=args.batch_size,
        device=args.device,
    )

    results = []

    print("Scoring fixed-k configurations with exact statistics...", flush=True)

    for k in args.fixed_k:
        settings = {"k": k}
        stats = load_exact_stats(root, "fixed", settings, latent_dim)

        results.append(evaluate_candidate_setting(
            candidate_values, candidate_ids, query_codes,
            candidates, qids, qrels,
            "fixed", settings, latent_dim, stats,
        ))

    print("Scoring adaptive configurations with exact statistics...", flush=True)

    for threshold in args.thresholds:
        for min_count in args.min_counts:
            for max_count in args.max_counts:
                if min_count > max_count:
                    continue

                settings = {
                    "threshold": threshold,
                    "min_concepts": min_count,
                    "max_concepts": max_count,
                }

                stats = load_exact_stats(
                    root, "adaptive", settings, latent_dim
                )

                results.append(evaluate_candidate_setting(
                    candidate_values, candidate_ids, query_codes,
                    candidates, qids, qrels,
                    "adaptive", settings, latent_dim, stats,
                ))

    fixed_rows = [
        row for row in results
        if row["selection"] == "fixed"
    ]

    if not fixed_rows:
        parser.error("provide at least one --fixed-k baseline")

    fixed_best = max(row["mrr_at_10"] for row in fixed_rows)

    eligible = [
        row for row in results
        if row["selection"] == "adaptive"
        and row["mrr_at_10"] >= fixed_best - 0.005
    ]

    recommendation = (
        min(
            eligible,
            key=lambda row: row["avg_passage_concepts_full_corpus"],
        )
        if eligible else None
    )

    report = {
        "scope": (
            "validation dense-candidate reranking proxy; "
            "full-corpus BM25 statistics; "
            "query encoding fixed at checkpoint threshold; "
            "confirm with full CL-SR search"
        ),
        "fixed_k_best_mrr_at_10": fixed_best,
        "query_encoding": "frozen checkpoint activation threshold",
        "query_threshold": query_threshold,
        "mrr_tolerance": 0.005,
        "statistics_scope": "full_corpus",
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
