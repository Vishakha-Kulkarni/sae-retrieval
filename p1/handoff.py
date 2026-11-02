"""Validate a frozen P2 artifact bundle and export P1 inputs without re-encoding text."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def query_ids(path, rows):
    ids = Path(path).read_text().split()
    if len(ids) != rows or len(set(ids)) != rows:
        raise ValueError(f'{path}: expected {rows} unique query IDs, got {len(ids)}')
    return ids


def validate_bundle(data_dir, split_dir, passage_path=None):
    root, split = Path(data_dir), Path(split_dir)
    passages = np.load(passage_path or root / 'passages.npy', mmap_mode='r')
    if passages.ndim != 2 or passages.shape[1] != 768 or not len(passages):
        raise ValueError('Expected nonempty [passages,768] SimLM embeddings')
    result = {'passage_shape': list(passages.shape), 'splits': {}}
    all_ids = {}
    for name, prefix in [('train', 'train'), ('validation', 'val')]:
        queries = np.load(root / f'{prefix}_queries.npy', mmap_mode='r')
        if queries.ndim != 2 or queries.shape[1] != passages.shape[1]:
            raise ValueError(f'{name}: embedding dimensions do not match passages')
        ids = query_ids(root / f'{prefix}_queries.ids', len(queries))
        expected = Path(split / f'{name}.qids').read_text().split()
        if len(expected) != len(set(expected)) or set(ids) != set(expected):
            raise ValueError(f'{name}: query IDs do not match the frozen shared split')
        from .metrics import read_qrels
        if set(read_qrels(split / f'{name}.qrels')) != set(ids):
            raise ValueError(f'{name}: qrels do not match query IDs')
        candidates = np.load(root / f'{prefix}.cands.npy', mmap_mode='r')
        teacher = np.load(root / f'{prefix}.teacher.npy', mmap_mode='r')
        if candidates.ndim != 2 or candidates.shape[0] != len(queries) or candidates.shape[1] < 2:
            raise ValueError(f'{name}: candidates must have shape [queries,K], K >= 2')
        if candidates.shape != teacher.shape or candidates.dtype.kind not in 'iu':
            raise ValueError(f'{name}: teacher shape or candidate dtype is invalid')
        for start in range(0, len(queries), 1024):
            rows = candidates[start:start + 1024]
            if rows.min() < 0 or rows.max() >= len(passages):
                raise ValueError(f'{name}: candidate passage row out of bounds')
            if np.any(np.diff(np.sort(rows, axis=1), axis=1) == 0):
                raise ValueError(f'{name}: duplicate candidates')
            if not np.isfinite(teacher[start:start + 1024]).all():
                raise ValueError(f'{name}: nonfinite teacher scores')
            if not np.isfinite(queries[start:start + 1024]).all():
                raise ValueError(f'{name}: nonfinite query embeddings')
        result['splits'][name] = {'queries': len(ids), 'candidates_per_query': candidates.shape[1]}
        all_ids[name] = set(ids)
    if all_ids['train'] & all_ids['validation']:
        raise ValueError('Training and validation query IDs overlap')
    return result


def load_baseline(path, threshold=None):
    from p2.sae import SAE
    # The HF bundle is external: require the restricted tensor/primitive loader.
    payload = torch.load(path, map_location='cpu', weights_only=True)
    state, meta = payload.get('state_dict', payload), payload.get('meta', {})
    if threshold is None:
        threshold = meta.get('threshold')
    if threshold is None or not math.isfinite(threshold) or threshold < 0:
        raise ValueError('Need this checkpoint\'s finite nonnegative threshold; do not reuse an older checkpoint\'s value')
    weight = state['linear.weight']
    if weight.ndim != 2 or weight.shape[0] % weight.shape[1]:
        raise ValueError('Unsupported SAE encoder dimensions')
    model = SAE(weight.shape[1], weight.shape[0] // weight.shape[1], meta.get('k', 32),
                decoder_norm=meta.get('args', {}).get('decoder_norm', 'row'))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) - {'steps_since_active'} or unexpected:
        raise ValueError(f'Incompatible SAE checkpoint: missing={missing}, unexpected={unexpected}')
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Checkpoint contains nonfinite weights')
    return model.eval(), float(threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, help='One compatible generation of P2 query/candidate/teacher files')
    parser.add_argument('--split-dir', required=True, help='P5 checksum-verified train/validation split')
    parser.add_argument('--passages', help='Passage .npy if stored outside data-dir')
    parser.add_argument('--checkpoint', required=True, help='Latest agreed B0 checkpoint, currently B0_final_atom.pt')
    parser.add_argument('--threshold', type=float, help='Only if absent from this checkpoint metadata')
    parser.add_argument('--output-dir', help='New output folder; omit to validate only')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    summary = validate_bundle(args.data_dir, args.split_dir, args.passages)
    model, threshold = load_baseline(args.checkpoint, args.threshold)
    if model.linear.weight.shape[1] != summary['passage_shape'][1]:
        raise ValueError('Checkpoint and embedding dimensions differ')
    from .train import file_hash
    summary.update(checkpoint_sha256=file_hash(args.checkpoint), threshold=threshold,
                   provenance='Shape/alignment checks passed; confirm all embeddings and teacher files belong to the same generation')
    print(json.dumps(summary, indent=2))
    if not args.output_dir:
        return
    from p2.evaluate import export
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    model.to(args.device)
    data = Path(args.data_dir)
    passage_path = Path(args.passages) if args.passages else data / 'passages.npy'
    export(model, threshold, np.load(passage_path, mmap_mode='r'), root / 'passages.latent.jsonl',
           device=args.device, batch_size=args.batch_size)
    for name, prefix in [('train', 'train'), ('validation', 'val')]:
        queries = np.load(data / f'{prefix}_queries.npy', mmap_mode='r')
        ids = query_ids(data / f'{prefix}_queries.ids', len(queries))
        export(model, threshold, queries, root / f'{name}.query.json', queries=True,
               ids=ids, device=args.device, batch_size=args.batch_size)
        candidates = np.load(data / f'{prefix}.cands.npy', mmap_mode='r')
        teacher = np.load(data / f'{prefix}.teacher.npy', mmap_mode='r')
        with open(root / f'{name}.run', 'w') as target:
            for qid, docs, scores in zip(ids, candidates, teacher):
                for rank, (docid, score) in enumerate(zip(docs, scores), 1):
                    target.write(f'{qid} Q0 {int(docid)} {rank} {float(score):.9g} simlm\n')
    (root / 'handoff.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
