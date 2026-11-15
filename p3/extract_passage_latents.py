"""Run P3 or fixed-k passage encoding with the released SAE implementation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .adaptive_encoder import encode_adaptive_batch, encode_fixed_k_batch
except ImportError:  # Also allow `python p3/extract_passage_latents.py`.
    from adaptive_encoder import encode_adaptive_batch, encode_fixed_k_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class NpyEmbeddingDataset(torch.utils.data.Dataset):
    """Read embeddings lazily from the float32 .npy files produced by P2."""

    def __init__(self, path: str):
        self.embeddings = np.load(path, mmap_mode="r")
        if self.embeddings.ndim != 2:
            raise ValueError(f"expected a 2D embedding matrix, got {self.embeddings.shape}")

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self.embeddings[index], copy=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-embeddings", "--input-embs-path", dest="input_embeddings", required=True,
                        help="Float32 .npy embedding matrix produced by p2.encode")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("adaptive", "fixed"), required=True)
    parser.add_argument("--threshold", type=float, help="Adaptive activation threshold")
    parser.add_argument("--min-concepts", type=int, default=8)
    parser.add_argument("--max-concepts", type=int, default=128)
    parser.add_argument("--k", type=int, help="Fixed-k baseline setting")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--input-dim", type=int, default=768)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-docs", type=int, help="Optional prefix limit for a small run")

    parser.add_argument(
    "--resume",
    action="store_true",
    help="Validate existing JSONL output and resume from the next document",
    )

    args = parser.parse_args()

    from p2.sae import load_sae

    if args.mode == "adaptive" and args.threshold is None:
        parser.error("--threshold is required for adaptive mode")
    if args.mode == "fixed" and args.k is None:
        parser.error("--k is required for fixed mode")

    dataset = NpyEmbeddingDataset(args.input_embeddings)
    total_docs = (
        min(len(dataset), args.max_docs)
        if args.max_docs is not None
        else len(dataset)
    )

    if args.max_docs is not None:
        dataset = torch.utils.data.Subset(dataset, range(total_docs))

    resume_count = 0

    if args.resume:
        if not os.path.isfile(args.output):
            parser.error(
                f"--resume requested but output file does not exist: {args.output}"
            )

        with open(args.output, "r", encoding="utf-8") as existing_file:
            for expected_docid, line in enumerate(existing_file):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    parser.error(
                        f"Invalid JSON on output line {expected_docid + 1}: {exc}"
                    )

                if not isinstance(record, dict):
                    parser.error(
                        f"Invalid record on output line {expected_docid + 1}: "
                        "expected a JSON object"
                    )

                if record.get("docid") != expected_docid:
                    parser.error(
                        f"Unexpected docid on line {expected_docid + 1}: "
                        f"expected {expected_docid}, got {record.get('docid')}"
                    )

                if not isinstance(record.get("ids"), list) or not isinstance(
                    record.get("weight"), list
                ):
                    parser.error(
                        f"Missing or invalid ids/weight on line {expected_docid + 1}"
                    )
                if len(record["ids"]) != len(record["weight"]):
                    parser.error(
                        f"IDs and weights differ in length on line {expected_docid + 1}"
                    )

                resume_count = expected_docid + 1

        if resume_count > total_docs:
            parser.error(
                f"Output contains {resume_count} records, "
                f"but input contains only {total_docs} documents"
            )

        dataset = torch.utils.data.Subset(
            dataset, range(resume_count, total_docs)
        )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=0, shuffle=False
    )


    model, _ = load_sae(args.checkpoint, map_location=args.device)
    if model.linear.in_features != args.input_dim:
        parser.error(f"checkpoint input dimension is {model.linear.in_features}, not {args.input_dim}")
    model = model.to(args.device)
    model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    encode = encode_adaptive_batch if args.mode == "adaptive" else encode_fixed_k_batch
    output_mode = "a" if args.resume else "w"

    with open(args.output, output_mode, encoding="utf-8") as output_file:
        doc_offset = resume_count
        for batch in tqdm(loader, desc=f"Encoding {args.mode} passages"):
            batch = batch.to(args.device)
            if args.mode == "adaptive":
                rows, _ = encode(
                    model,
                    batch,
                    threshold=args.threshold,
                    min_concepts=args.min_concepts,
                    max_concepts=args.max_concepts,
                    return_reconstruction=False,
                )
            else:
                rows, _ = encode(
                    model,
                    batch,
                    k=args.k,
                    return_reconstruction=False,
                )
            for row_idx, ids, weights in rows:
                output_file.write(json.dumps({
                    "docid": doc_offset + row_idx,
                    "ids": ids,
                    "weight": weights,
                }) + "\n")
            doc_offset += len(batch)


if __name__ == "__main__":
    main()
