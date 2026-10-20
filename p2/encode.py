"""Shared week-1 task: encode MS MARCO with SimLM into float32 .npy files (row i = passage id i).

python -m p2.encode passages --input collection.tsv --output passages.npy
python -m p2.encode queries  --input queries.train.tsv --output train_queries.npy
Matches upstream SimLM inference: CLS pooling, no L2 norm, q_max_len 32, p_max_len 144,
passages encoded as the (title, contents) pair with title '-' when missing.
"""
import argparse
import gzip
from pathlib import Path

import numpy as np
import torch

MODEL = 'intfloat/simlm-base-msmarco-finetuned'


def read_tsv(path, limit=None):
    opener = gzip.open if str(path).endswith('.gz') else open
    ids, texts = [], []
    with opener(path, 'rt') as source:
        for line in source:
            if limit is not None and len(ids) >= limit:
                break
            parts = line.rstrip('\n').split('\t')
            ids.append(parts[0])
            texts.append(parts[1:])
    return ids, texts


def default_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


@torch.no_grad()
def encode(texts, kind, output, batch_size=256, device=None, model_name=MODEL):
    from transformers import AutoModel, AutoTokenizer
    device = device or default_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if device == 'cuda':
        model.half()
    out = np.lib.format.open_memmap(output, mode='w+', dtype=np.float32, shape=(len(texts), 768))
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        if kind == 'passages':
            titles = [t[0] if len(t) > 1 and t[0] else '-' for t in chunk]
            bodies = [t[-1] for t in chunk]
            batch = tokenizer(titles, bodies, max_length=144, truncation='only_second',
                              padding=True, return_tensors='pt')
        else:
            batch = tokenizer([t[0] for t in chunk], max_length=32, truncation=True,
                              padding=True, return_tensors='pt')
        hidden = model(**batch.to(device)).last_hidden_state[:, 0]
        out[start:start + len(chunk)] = hidden.float().cpu().numpy()
        if start // batch_size % 200 == 0:
            print(f'{start + len(chunk)}/{len(texts)}', flush=True)
    out.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('kind', choices=['passages', 'queries'])
    parser.add_argument('--input', required=True, help='collection.tsv[.gz] (id, [title,] text) or queries TSV')
    parser.add_argument('--output', required=True, help='.npy; for queries a sibling .ids file is written')
    parser.add_argument('--limit', type=int, help='First N rows (subset runs only)')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device')
    args = parser.parse_args()
    ids, texts = read_tsv(args.input, args.limit)
    if args.kind == 'passages' and any(str(i) != d for i, d in enumerate(ids)):
        raise ValueError('Passage IDs must equal row positions, as SimLM/CL-SR assume')
    encode(texts, args.kind, args.output, args.batch_size, args.device)
    Path(args.output).with_suffix('.ids').write_text(''.join(i + '\n' for i in ids))


if __name__ == '__main__':
    main()
