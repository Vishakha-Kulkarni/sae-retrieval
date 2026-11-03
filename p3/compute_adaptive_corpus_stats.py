import argparse
import json
from pathlib import Path

import numpy as np


def compute_stats(ids, values, threshold, min_concepts, chunk_size):
    num_docs, max_concepts = ids.shape
    latent_dim = 24576

    df = np.zeros(latent_dim, dtype=np.uint32)
    total_length = 0

    rows = np.arange(min(num_docs, chunk_size))

    for start in range(0, num_docs, chunk_size):
        end = min(start + chunk_size, num_docs)

        block_ids = np.asarray(ids[start:end])
        block_values = np.asarray(values[start:end])

        positive_counts = np.sum(block_values > 0, axis=1)
        threshold_counts = np.sum(block_values > threshold, axis=1)

        fallback_counts = np.minimum(min_concepts, positive_counts)
        selected_counts = np.maximum(threshold_counts, fallback_counts)

        # Selection is a prefix because activations are sorted descending.
        positions = np.arange(max_concepts)[None, :]
        selected_mask = positions < selected_counts[:, None]

        selected_ids = block_ids[selected_mask]

        if selected_ids.size:
            if selected_ids.max() >= latent_dim:
                raise ValueError("Latent ID outside latent dimension")

            df += np.bincount(
                selected_ids.astype(np.int64),
                minlength=latent_dim,
            ).astype(np.uint32)

        total_length += int(selected_counts.sum())

        if start == 0 or end == num_docs or end % 500000 < chunk_size:
            print(
                f"  Processed {end:,}/{num_docs:,} passages",
                flush=True,
            )

    return {
        "num_docs": num_docs,
        "avg_doc_length": total_length / num_docs,
        "document_frequency": df,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--thresholds",
        default="0.35,0.45,0.49723,0.55,0.65,0.80",
    )
    parser.add_argument("--min-concepts", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=25000)
    args = parser.parse_args()

    cache = Path(args.cache)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((cache / "metadata.json").read_text())
    if not metadata.get("completed"):
        raise RuntimeError("Top-128 cache is not complete")

    ids = np.load(cache / "ids.npy", mmap_mode="r")
    values = np.load(cache / "values.npy", mmap_mode="r")

    if ids.shape != values.shape or ids.ndim != 2:
        raise ValueError("Unexpected cache shapes")

    num_docs, max_concepts = ids.shape
    latent_dim = int(metadata["config"]["latent_dim"])

    if num_docs != metadata["config"]["num_passages"]:
        raise ValueError("Passage count does not match cache metadata")

    if max_concepts != 128:
        raise ValueError(f"Expected top-128 cache, got {max_concepts}")

    thresholds = [float(x) for x in args.thresholds.split(",")]

    print(f"Passages: {num_docs:,}")
    print(f"Latent dimension: {latent_dim:,}")
    print(f"Minimum concepts: {args.min_concepts}")
    print(f"Thresholds: {thresholds}")

    for threshold in thresholds:
        name = f"threshold_{threshold:.5f}"
        target = output / name

        if target.exists():
            raise FileExistsError(
                f"{target} already exists. Choose a new output directory."
            )

        print(f"\nComputing exact statistics for threshold={threshold}")

        stats = compute_stats(
            ids,
            values,
            threshold,
            args.min_concepts,
            args.chunk_size,
        )

        target.mkdir()

        np.save(target / "document_frequency.npy",
                stats["document_frequency"])

        result_metadata = {
            "threshold": threshold,
            "min_concepts": args.min_concepts,
            "max_concepts": max_concepts,
            "num_docs": stats["num_docs"],
            "avg_doc_length": stats["avg_doc_length"],
            "latent_dim": latent_dim,
            "statistics_scope": "full_corpus",
            "selection": "activation_threshold_with_minimum_fallback",
            "cache": str(cache.resolve()),
        }

        (target / "metadata.json").write_text(
            json.dumps(result_metadata, indent=2) + "\n"
        )

        print(f"  Average concepts: {stats['avg_doc_length']:.6f}")
        print(f"  Saved: {target}")

    print("\nAll thresholds completed.")


if __name__ == "__main__":
    main()
