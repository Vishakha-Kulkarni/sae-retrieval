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
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-docs", type=int, default=50000)
    args = parser.parse_args()

    cache = Path(args.cache)
    output = Path(args.output)
    building = output.with_name(output.name + ".building")

    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if building.exists():
        raise FileExistsError(
            f"Temporary directory already exists: {building}. "
            "Inspect it before removing it."
        )

    ids = np.load(cache / "ids.npy", mmap_mode="r")
    values = np.load(cache / "values.npy", mmap_mode="r")
    metadata = json.loads((cache / "metadata.json").read_text())

    if not metadata.get("completed"):
        raise RuntimeError("Top128 cache is not marked complete")
    if ids.shape != values.shape or ids.shape[1] < 128:
        raise ValueError(f"Unexpected cache shapes: {ids.shape}, {values.shape}")

    num_docs = ids.shape[0]
    latent_dim = int(metadata["config"]["latent_dim"])
    k = 128
    shard_count = 256
    terms_per_shard = (latent_dim + shard_count - 1) // shard_count

    if latent_dim > 65536:
        raise ValueError("uint16 term IDs cannot represent this latent dimension")

    building.mkdir(parents=True)
    handles = []
    doc_lengths = np.zeros(num_docs, dtype=np.uint8)

    try:
        print(f"Building fixed128 disk index: {num_docs:,} documents", flush=True)
        print(f"Concepts: {latent_dim:,}; shards: {shard_count}", flush=True)

        # Open one buffered output file per concept shard.
        for shard in range(shard_count):
            path = building / f"shard_{shard:03d}.bin"
            handles.append(open(path, "wb", buffering=8 * 1024 * 1024))

        for start in tqdm(range(0, num_docs, args.chunk_docs), desc="Writing postings"):
            end = min(start + args.chunk_docs, num_docs)

            block_ids = np.asarray(ids[start:end, :k])
            block_values = np.asarray(values[start:end, :k])
            mask = block_values > 0

            row_idx, col_idx = np.nonzero(mask)
            if len(row_idx) == 0:
                continue

            terms = block_ids[row_idx, col_idx].astype(np.uint16, copy=False)
            weights = block_values[row_idx, col_idx].astype(np.float32, copy=False)
            docids = (row_idx + start).astype(np.uint32, copy=False)

            if np.any(terms >= latent_dim):
                raise ValueError("Cache contains a concept ID outside latent_dim")

            lengths = np.bincount(row_idx, minlength=end - start)
            if lengths.max(initial=0) > 128:
                raise ValueError("A document has more than 128 selected concepts")
            doc_lengths[start:end] = lengths.astype(np.uint8)

            records = np.empty(len(terms), dtype=RECORD_DTYPE)
            records["term"] = terms
            records["docid"] = docids
            records["weight"] = weights

            # Group records by shard before writing.
            shard_ids = terms.astype(np.uint32) // terms_per_shard
            order = np.argsort(shard_ids, kind="stable")
            records = records[order]
            sorted_shards = shard_ids[order]

            boundaries = np.flatnonzero(sorted_shards[1:] != sorted_shards[:-1]) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [len(records)]))

            for lo, hi in zip(starts, ends):
                shard = int(sorted_shards[lo])
                records[lo:hi].tofile(handles[shard])

        for handle in handles:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        handles.clear()

        # Store compact document lengths.
        np.save(building / "document_lengths.npy", doc_lengths)

        # Sort each shard by (term, docid), then compute document frequencies.
        term_stats = np.zeros(
            latent_dim,
            dtype=[("doc_freq", "<u4"), ("idf", "<f8")],
        )

        for shard in tqdm(range(shard_count), desc="Sorting shards"):
            path = building / f"shard_{shard:03d}.bin"
            size = path.stat().st_size

            if size == 0:
                continue
            if size % RECORD_DTYPE.itemsize:
                raise RuntimeError(f"Corrupt shard file: {path}")

            records = np.memmap(path, dtype=RECORD_DTYPE, mode="r+")
            records.sort(order=["term", "docid"], kind="quicksort")
            records.flush()

            unique_terms, counts = np.unique(
                records["term"], return_counts=True
            )
            for term, count in zip(unique_terms, counts):
                term = int(term)
                count = int(count)
                term_stats[term]["doc_freq"] = count
                term_stats[term]["idf"] = math.log(
                    (num_docs - count + 0.5) / (count + 0.5)
                )

            del records

        np.save(building / "term_stats.npy", term_stats)

        avgdl = float(doc_lengths.astype(np.uint64).sum()) / num_docs
        index_metadata = {
            "format": "fixed128_disk_v1",
            "num_docs": int(num_docs),
            "latent_dim": int(latent_dim),
            "k": 128,
            "shard_count": shard_count,
            "terms_per_shard": terms_per_shard,
            "avgdl": avgdl,
            "record_dtype": "term:uint16,docid:uint32,weight:float32",
            "record_bytes": RECORD_DTYPE.itemsize,
            "completed": True,
        }

        (building / "metadata.json").write_text(
            json.dumps(index_metadata, indent=2) + "\n"
        )

        # Confirm all shards are structurally valid before publishing.
        for shard in range(shard_count):
            path = building / f"shard_{shard:03d}.bin"
            if path.stat().st_size % RECORD_DTYPE.itemsize:
                raise RuntimeError(f"Invalid shard size: {path}")

        os.replace(building, output)

        total_bytes = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
        print("[Done] Fixed128 disk index created", flush=True)
        print(f"Path: {output}", flush=True)
        print(f"Documents: {num_docs:,}", flush=True)
        print(f"Average document length: {avgdl:.6f}", flush=True)
        print(f"Index directory size: {total_bytes / 1024**3:.2f} GiB", flush=True)

    except Exception:
        for handle in handles:
            try:
                handle.close()
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
