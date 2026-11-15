from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

RECORD_DTYPE = np.dtype([
    ("term", "<u2"),
    ("docid", "<u4"),
    ("weight", "<f4"),
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--min-concepts", type=int, default=8)
    parser.add_argument("--max-concepts", type=int, default=128)
    parser.add_argument("--chunk-docs", type=int, default=50000)
    args = parser.parse_args()

    cache = Path(args.cache)
    stats_dir = Path(args.stats)
    output = Path(args.output)
    building = output.with_name(output.name + ".building")

    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if building.exists():
        raise FileExistsError(f"Temporary directory already exists: {building}")

    if not math.isfinite(args.threshold) or args.threshold < 0:
        raise ValueError("Threshold must be finite and nonnegative")
    if args.min_concepts < 1 or args.max_concepts < args.min_concepts:
        raise ValueError("Invalid min/max concepts")

    ids = np.load(cache / "ids.npy", mmap_mode="r")
    values = np.load(cache / "values.npy", mmap_mode="r")
    cache_meta = json.loads((cache / "metadata.json").read_text())

    if not cache_meta.get("completed"):
        raise RuntimeError("Top128 cache is incomplete")
    if ids.shape != values.shape or ids.ndim != 2:
        raise ValueError("Invalid cache shapes")
    if ids.shape[1] < args.max_concepts:
        raise ValueError("Cache does not contain enough concepts")

    num_docs = ids.shape[0]
    latent_dim = int(cache_meta["config"]["latent_dim"])

    if latent_dim > 65536:
        raise ValueError("uint16 cannot represent latent dimension")

    stats_meta = json.loads((stats_dir / "metadata.json").read_text())
    if stats_meta.get("statistics_scope") != "full_corpus":
        raise RuntimeError("Expected full-corpus statistics")
    if int(stats_meta["num_docs"]) != num_docs:
        raise RuntimeError("Statistics corpus size does not match cache")
    if int(stats_meta["latent_dim"]) != latent_dim:
        raise RuntimeError("Statistics latent dimension mismatch")
    if int(stats_meta["min_concepts"]) != args.min_concepts:
        raise RuntimeError("Statistics min_concepts mismatch")
    if int(stats_meta["max_concepts"]) != args.max_concepts:
        raise RuntimeError("Statistics max_concepts mismatch")
    if not math.isclose(
        float(stats_meta["threshold"]), args.threshold,
        rel_tol=0.0, abs_tol=1e-7
    ):
        raise RuntimeError("Statistics threshold mismatch")

    df = np.load(stats_dir / "document_frequency.npy", mmap_mode="r")
    if df.shape != (latent_dim,):
        raise RuntimeError("Invalid document-frequency array")

    avgdl = float(stats_meta["avg_doc_length"])

    shard_count = 256
    terms_per_shard = (latent_dim + shard_count - 1) // shard_count

    building.mkdir(parents=True)
    handles = []
    doc_lengths = np.zeros(num_docs, dtype=np.uint8)
    total_records = 0

    try:
        print(
            f"Building adaptive index: {num_docs:,} documents",
            flush=True,
        )
        print(
            f"Threshold={args.threshold}, min={args.min_concepts}, "
            f"max={args.max_concepts}",
            flush=True,
        )

        for shard in range(shard_count):
            path = building / f"shard_{shard:03d}.bin"
            handles.append(open(path, "wb", buffering=8 * 1024 * 1024))

        for start in tqdm(
            range(0, num_docs, args.chunk_docs),
            desc="Writing adaptive postings",
        ):
            end = min(start + args.chunk_docs, num_docs)

            block_ids = np.asarray(ids[start:end, :args.max_concepts])
            block_values = np.asarray(values[start:end, :args.max_concepts])

            positive_counts = np.sum(block_values > 0, axis=1)
            threshold_counts = np.sum(
                block_values > args.threshold, axis=1
            )

            fallback_counts = np.minimum(
                positive_counts, args.min_concepts
            )
            lengths = np.maximum(threshold_counts, fallback_counts)
            lengths = np.minimum(lengths, args.max_concepts)

            doc_lengths[start:end] = lengths.astype(np.uint8)

            row_idx = np.repeat(
                np.arange(end - start, dtype=np.int64),
                lengths.astype(np.int64),
            )
            if len(row_idx) == 0:
                continue

            offsets = np.arange(len(row_idx), dtype=np.int64)
            row_starts = np.repeat(
                np.cumsum(lengths, dtype=np.int64) - lengths,
                lengths.astype(np.int64),
            )
            col_idx = offsets - row_starts

            terms = block_ids[row_idx, col_idx].astype(
                np.uint16, copy=False
            )
            weights = block_values[row_idx, col_idx].astype(
                np.float32, copy=False
            )
            docids = (row_idx + start).astype(
                np.uint32, copy=False
            )

            if np.any(terms >= latent_dim):
                raise ValueError("Concept ID outside latent dimension")
            if np.any(weights <= 0):
                raise ValueError("Selected non-positive activation")

            records = np.empty(len(terms), dtype=RECORD_DTYPE)
            records["term"] = terms
            records["docid"] = docids
            records["weight"] = weights

            shard_ids = terms.astype(np.uint32) // terms_per_shard

            # Sort by shard, then term, then document ID.
            order = np.lexsort((
                records["docid"],
                records["term"],
                shard_ids,
            ))

            records = records[order]
            sorted_shards = shard_ids[order]

            boundaries = np.flatnonzero(
                sorted_shards[1:] != sorted_shards[:-1]
            ) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [len(records)]))

            for lo, hi in zip(starts, ends):
                shard = int(sorted_shards[lo])
                records[lo:hi].tofile(handles[shard])

            total_records += len(records)

        for handle in handles:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        handles.clear()

        np.save(building / "document_lengths.npy", doc_lengths)

        term_stats = np.zeros(
            latent_dim,
            dtype=[("doc_freq", "<u4"), ("idf", "<f8")],
        )

        term_stats["doc_freq"] = df
        term_stats["idf"] = np.log(
            (num_docs - df.astype(np.float64) + 0.5)
            / (df.astype(np.float64) + 0.5)
        )

        np.save(building / "term_stats.npy", term_stats)

        actual_avgdl = float(
            doc_lengths.astype(np.uint64).sum()
        ) / num_docs

        if not math.isclose(
            actual_avgdl, avgdl, rel_tol=0.0, abs_tol=1e-6
        ):
            raise RuntimeError(
                f"Document length mismatch: built={actual_avgdl}, "
                f"stats={avgdl}"
            )

        index_metadata = {
            "format": "fixed_disk_v1",
            "selection": "activation_threshold_with_minimum_fallback",
            "threshold": args.threshold,
            "min_concepts": args.min_concepts,
            "max_concepts": args.max_concepts,
            "num_docs": int(num_docs),
            "latent_dim": latent_dim,
            "k": args.max_concepts,
            "shard_count": shard_count,
            "terms_per_shard": terms_per_shard,
            "avgdl": actual_avgdl,
            "total_postings": int(total_records),
            "statistics_scope": "full_corpus",
            "record_dtype": "term:uint16,docid:uint32,weight:float32",
            "record_bytes": RECORD_DTYPE.itemsize,
            "completed": True,
        }

        (building / "metadata.json").write_text(
            json.dumps(index_metadata, indent=2) + "\n"
        )

        os.replace(building, output)

        total_bytes = sum(
            p.stat().st_size for p in output.iterdir()
            if p.is_file()
        )

        print("[Done] Adaptive disk index created", flush=True)
        print(f"Path: {output}", flush=True)
        print(f"Average document length: {actual_avgdl:.6f}", flush=True)
        print(f"Total postings: {total_records:,}", flush=True)
        print(f"Index size: {total_bytes / 1024**3:.2f} GiB", flush=True)

    except Exception:
        for handle in handles:
            try:
                handle.close()
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
