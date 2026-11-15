import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from p2.sae import load_sae


def atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-embeddings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Refusing to run extraction on CPU.")

    device = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    embeddings = np.load(args.input_embeddings, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {embeddings.shape}")

    n_passages, input_dim = embeddings.shape
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ids_path = output_dir / "ids.npy"
    values_path = output_dir / "values.npy"
    progress_path = output_dir / "checkpoint.json"

    model, meta = load_sae(args.checkpoint, k=32, map_location="cpu")
    model.to(device).eval()

    latent_dim = model.linear.weight.shape[0]
    if model.linear.weight.shape[1] != input_dim:
        raise ValueError("Embedding dimension does not match SAE input dimension.")
    if args.k > latent_dim:
        raise ValueError(f"k={args.k} exceeds latent dimension {latent_dim}")
    if latent_dim > np.iinfo(np.uint16).max:
        raise ValueError("Latent dimension is too large for uint16 IDs.")

    config = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "input_embeddings": str(Path(args.input_embeddings).resolve()),
        "num_passages": n_passages,
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "k": args.k,
        "ids_dtype": "uint16",
        "values_dtype": "float32",
    }

    if args.resume:
        if not progress_path.exists() or not ids_path.exists() or not values_path.exists():
            raise FileNotFoundError("Cannot resume: cache files or checkpoint.json are missing.")

        with open(progress_path) as f:
            progress = json.load(f)

        if progress["config"] != config:
            raise ValueError("Cache configuration does not match this run.")

        ids = np.load(ids_path, mmap_mode="r+")
        values = np.load(values_path, mmap_mode="r+")
        start = int(progress["next_passage"])

        if ids.shape != (n_passages, args.k) or values.shape != (n_passages, args.k):
            raise ValueError("Cache array shapes do not match expected dimensions.")
    else:
        if ids_path.exists() or values_path.exists() or progress_path.exists():
            raise FileExistsError(
                f"Cache already exists in {output_dir}. Use --resume or choose a new directory."
            )

        ids = np.lib.format.open_memmap(
            ids_path, mode="w+", dtype=np.uint16,
            shape=(n_passages, args.k)
        )
        values = np.lib.format.open_memmap(
            values_path, mode="w+", dtype=np.float32,
            shape=(n_passages, args.k)
        )
        start = 0

        ids.flush()
        values.flush()
        atomic_json(progress_path, {
            "config": config,
            "next_passage": 0,
        })

    print(f"Passages: {n_passages:,}", flush=True)
    print(f"Latent dimension: {latent_dim:,}", flush=True)
    print(f"Top-k: {args.k}", flush=True)
    print(f"Starting at passage: {start:,}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)

    with torch.inference_mode():
        for batch_start in range(start, n_passages, args.batch_size):
            batch_end = min(batch_start + args.batch_size, n_passages)

            batch_np = np.array(embeddings[batch_start:batch_end], dtype=np.float32)
            batch = torch.from_numpy(batch_np).to(device)

            activations = model.linear(batch - model.bias_pre)
            top_values, top_ids = torch.topk(
                activations,
                k=args.k,
                dim=1,
                largest=True,
                sorted=True,
            )
            top_values = top_values.clamp_min(0)

            ids[batch_start:batch_end] = top_ids.to(
                torch.uint16
            ).cpu().numpy()
            values[batch_start:batch_end] = top_values.float().cpu().numpy()

            # Flush data before advancing the resume checkpoint.
            ids.flush()
            values.flush()

            atomic_json(progress_path, {
                "config": config,
                "next_passage": batch_end,
            })

            if batch_end % 50000 < args.batch_size or batch_end == n_passages:
                print(
                    f"Processed {batch_end:,}/{n_passages:,} passages",
                    flush=True,
                )

            del batch, activations, top_values, top_ids

    atomic_json(output_dir / "metadata.json", {
        "config": config,
        "completed": True,
    })

    print("Top-k cache extraction completed.", flush=True)


if __name__ == "__main__":
    main()
