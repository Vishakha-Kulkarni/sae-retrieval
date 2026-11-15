from pathlib import Path
import argparse
import json
import numpy as np
from tqdm import tqdm

RECORD_DTYPE = np.dtype([
    ("term", "<u2"),
    ("docid", "<u4"),
    ("weight", "<f4"),
])

def sort_shards(index_dir, shard_count=256, chunk_records=5_000_000):
    index_dir = Path(index_dir)

    for shard_id in tqdm(range(shard_count), desc="Sorting shards"):
        path = index_dir / f"shard_{shard_id:03d}.bin"
        if path.stat().st_size == 0:
            continue

        records = np.memmap(path, dtype=RECORD_DTYPE, mode="r")
        n = len(records)

        # Sort this shard in bounded-memory chunks first.
        runs_dir = index_dir / f".sort_{shard_id:03d}"
        runs_dir.mkdir()

        run_paths = []

        for start in range(0, n, chunk_records):
            end = min(start + chunk_records, n)
            chunk = np.array(records[start:end])

            order = np.lexsort((
                chunk["docid"],
                chunk["term"],
            ))
            chunk = chunk[order]

            run_path = runs_dir / f"run_{len(run_paths):04d}.bin"
            chunk.tofile(run_path)
            run_paths.append(run_path)

        del records

        # Merge sorted runs by (term, docid).
        import heapq

        runs = [
            np.memmap(p, dtype=RECORD_DTYPE, mode="r")
            for p in run_paths
        ]

        positions = [0] * len(runs)
        heap = []

        for i, run in enumerate(runs):
            if len(run):
                r = run[0]
                heapq.heappush(
                    heap,
                    (int(r["term"]), int(r["docid"]), i)
                )

        temp_path = path.with_suffix(".sorted.tmp")

        with open(temp_path, "wb", buffering=8 * 1024 * 1024) as out:
            buffer = np.empty(1_000_000, dtype=RECORD_DTYPE)
            used = 0

            while heap:
                term, docid, i = heapq.heappop(heap)
                buffer[used] = runs[i][positions[i]]
                used += 1

                if used == len(buffer):
                    buffer.tofile(out)
                    used = 0

                positions[i] += 1

                if positions[i] < len(runs[i]):
                    r = runs[i][positions[i]]
                    heapq.heappush(
                        heap,
                        (int(r["term"]), int(r["docid"]), i)
                    )

            if used:
                buffer[:used].tofile(out)

        del runs
        del heap

        temp_path.replace(path)

        for p in run_paths:
            p.unlink()
        runs_dir.rmdir()

    print("All shards globally sorted.")


def main():
    parser = argparse.ArgumentParser(
        description="Externally sort adaptive posting shards by term and document ID."
    )
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--shard-count", type=int, default=256)
    parser.add_argument("--chunk-records", type=int, default=5_000_000)
    args = parser.parse_args()

    if args.shard_count < 1 or args.chunk_records < 1:
        parser.error("--shard-count and --chunk-records must be positive")

    index_dir = Path(args.index_dir)
    metadata_path = index_dir / "metadata.json"
    if not metadata_path.is_file():
        parser.error(f"Index metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("completed"):
        parser.error(f"Index is not marked complete: {index_dir}")
    if int(metadata.get("shard_count", -1)) != args.shard_count:
        parser.error(
            f"Requested {args.shard_count} shards but index metadata declares "
            f"{metadata.get('shard_count')}"
        )

    sort_shards(
        index_dir,
        shard_count=args.shard_count,
        chunk_records=args.chunk_records,
    )


if __name__ == "__main__":
    main()
