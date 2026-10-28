"""Build and evaluate a full CL-SR index for a P3 latent JSONL file.

Uses the pinned upstream index builder and BM25 scoring function. The input
query latent file must be in the released ``{qid: {ids, weight}}`` JSON schema.
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
import tempfile
import time
from pathlib import Path

from p1.metrics import evaluate, read_qrels


ROOT = Path(__file__).resolve().parents[1]
CLSR = ROOT / "upstream" / "Decoding-Dense-Embeddings" / "cl-sr"
sys.path.insert(0, str(CLSR))


def _validate_records(path: Path) -> tuple[int, float]:
    import json as json_module

    count = 0
    total_lengths = 0
    seen_docids = set()
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json_module.loads(line)
                docid, ids, weights = row["docid"], row["ids"], row["weight"]
            except (json_module.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid passage latent JSONL at line {line_no}") from exc
            if len(ids) != len(weights) or len(set(ids)) != len(ids):
                raise ValueError(f"line {line_no}: ids and weights must be aligned with unique IDs")
            if not isinstance(docid, (str, int)):
                raise ValueError(f"line {line_no}: docid must be a string or integer")
            docid = str(docid)
            if docid in seen_docids:
                raise ValueError(f"line {line_no}: duplicate docid {docid!r}")
            seen_docids.add(docid)
            if any(not isinstance(x, (int, float)) or x <= 0 for x in weights):
                raise ValueError(f"line {line_no}: all concept weights must be positive numbers")
            count += 1
            total_lengths += len(ids)
    if not count:
        raise ValueError("passage latent JSONL is empty")
    return count, total_lengths / count


def run_search(index, query_path: Path, qrels, top_k: int = 1000):
    import latent_retrieve

    latent_retrieve.TOP_K = top_k
    with query_path.open(encoding="utf-8") as source:
        queries = json.load(source)
    if not isinstance(queries, dict) or not queries:
        raise ValueError("query latent JSON must be a non-empty object")
    run = {}
    latencies_ms = []
    for qid, values in queries.items():
        ids, weights = values["ids"], values["weight"]
        if len(ids) != len(weights):
            raise ValueError(f"query {qid}: ids and weights are not aligned")
        start = time.perf_counter()
        hits = latent_retrieve.retrieve_one(
            [str(concept) for concept in ids], weights,
            index["inverted_index"], index["document_lengths"], index["avgdl"],
        )
        latencies_ms.append((time.perf_counter() - start) * 1000)
        run[str(qid)] = [str(docid) for docid, _ in hits]
    relevant_qrels = {qid: judgments for qid, judgments in qrels.items() if qid in run}
    if not relevant_qrels:
        raise ValueError("query latent IDs and qrels have no matching query IDs")
    return evaluate(run, relevant_qrels), latencies_ms, len(relevant_qrels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passage-latents", required=True)
    parser.add_argument("--query-latents", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--output", help="Optional path for the JSON metrics report")
    parser.add_argument("--max-latents", type=int,
                        help="Optional upstream truncation; normally leave unset for adaptive files")
    args = parser.parse_args()

    from indexing import build_inverted_index

    passage_path = Path(args.passage_latents)
    query_path = Path(args.query_latents)
    if not passage_path.is_file() or not query_path.is_file():
        parser.error("passage and query latent files must exist")
    if args.max_latents is not None and args.max_latents < 1:
        parser.error("--max-latents must be positive")
    docs, mean_concepts = _validate_records(passage_path)
    qrels = read_qrels(args.qrels)
    with tempfile.TemporaryDirectory(prefix="p3-clsr-") as directory:
        index_path = Path(directory) / "index.pkl"
        start = time.perf_counter()
        build_inverted_index(passage_path, index_path, max_latent_num=args.max_latents)
        index_build_s = time.perf_counter() - start
        index_size_bytes = index_path.stat().st_size
        with index_path.open("rb") as source:
            index = pickle.load(source)
        metrics, latencies, num_evaluated_queries = run_search(index, query_path, qrels)
        if args.max_latents is None:
            indexed_mean_concepts = index["avgdl"]
        else:
            indexed_mean_concepts = sum(index["document_lengths"].values()) / docs

    report = {
        "passage_latents": str(passage_path),
        "query_latents": str(query_path),
        "num_passages": docs,
        "avg_concepts_per_passage": indexed_mean_concepts,
        "index_size_bytes": index_size_bytes,
        "index_size_gib": index_size_bytes / (1024 ** 3),
        "index_build_seconds": index_build_s,
        "mean_query_latency_ms": statistics.mean(latencies),
        "median_query_latency_ms": statistics.median(latencies),
        "num_timed_queries": len(latencies),
        "num_evaluated_queries": num_evaluated_queries,
        **metrics,
    }
    serialized = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
