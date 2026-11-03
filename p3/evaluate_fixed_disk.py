from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from p1.metrics import evaluate, read_qrels
from p3.fixed_disk import FixedDiskIndex


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def directory_size(path):
    return sum(
        p.stat().st_size
        for p in Path(path).rglob("*")
        if p.is_file()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--every", type=int, default=5)
    args = parser.parse_args()

    print("Opening fixed-k disk index...", flush=True)
    index = FixedDiskIndex(args.index)

    print(
        f"Index ready: k={index.k}; {index.total_docs:,} documents; "
        f"avgdl={index.avgdl:.6f}",
        flush=True,
    )

    with open(args.queries, encoding="utf-8") as f:
        queries = json.load(f)

    qrels = read_qrels(args.qrels)
    checkpoint_path = Path(args.checkpoint)

    if checkpoint_path.exists():
        checkpoint = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )
        run = checkpoint["run"]
        latencies = checkpoint["latencies_ms"]
        completed = set(checkpoint["completed"])

        print(
            f"Resuming: {len(completed)}/{len(queries)} queries done",
            flush=True,
        )
    else:
        run, latencies, completed = {}, [], set()

    start_all = time.perf_counter()
    total = len(queries)

    for qid, values in queries.items():
        qid = str(qid)

        if qid in completed:
            continue

        ids, weights = values["ids"], values["weight"]

        if len(ids) != len(weights):
            raise ValueError(f"Query {qid}: IDs and weights differ in length")

        start = time.perf_counter()

        hits = index.retrieve_one(
            [int(x) for x in ids],
            weights,
            top_k=1000,
        )

        latencies.append((time.perf_counter() - start) * 1000)
        run[qid] = [str(docid) for docid, _ in hits]
        completed.add(qid)

        if len(completed) % args.every == 0 or len(completed) == total:
            atomic_json(args.checkpoint, {
                "run": run,
                "latencies_ms": latencies,
                "completed": sorted(completed),
            })

            elapsed = time.perf_counter() - start_all
            print(
                f"Progress: {len(completed)}/{total} "
                f"({100 * len(completed) / total:.1f}%), "
                f"elapsed this run: {elapsed / 60:.1f} min",
                flush=True,
            )

    relevant_qrels = {
        str(q): v for q, v in qrels.items() if str(q) in run
    }

    if not relevant_qrels:
        raise ValueError("No query IDs match the qrels")

    metrics = evaluate(run, relevant_qrels)
    size_bytes = directory_size(args.index)

    report = {
        "index_path": str(Path(args.index).resolve()),
        "k": index.k,
        "num_passages": index.total_docs,
        "avg_concepts_per_passage": index.avgdl,
        "index_size_bytes": size_bytes,
        "index_size_gib": size_bytes / (1024 ** 3),
        "index_build_seconds": None,
        "mean_query_latency_ms": statistics.mean(latencies),
        "median_query_latency_ms": statistics.median(latencies),
        "num_timed_queries": len(latencies),
        "num_evaluated_queries": len(relevant_qrels),
        **metrics,
    }

    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
