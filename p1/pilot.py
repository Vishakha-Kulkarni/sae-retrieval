"""Small real-data reranking pilot using frozen B0 activations and corrected teachers."""
import argparse
import hashlib
from collections import Counter
import json
from pathlib import Path
import random
import subprocess
import sys

import ijson
import numpy as np
import torch

from .handoff import load_baseline, query_ids
from .metrics import read_qrels
from .prepare import latent
from .train import file_hash


def prepare(bundle, output, train_limit=256, validation_limit=256, max_concepts=24, seed=42):
    bundle, output = Path(bundle), Path(output)
    if min(train_limit, validation_limit, max_concepts) < 1:
        raise ValueError('Query limits and max-concepts must be positive')
    data = bundle / 'data_titled'
    selected, candidates, scores = {}, {}, {}
    for name, prefix, limit in [('train', 'train', train_limit), ('validation', 'val', validation_limit)]:
        ids = (data / f'{prefix}_queries.ids').read_text().split()
        rows = np.load(data / f'{prefix}.cands.npy', mmap_mode='r')
        teacher = np.load(data / f'{prefix}.teacher.npy', mmap_mode='r')
        query_ids(data / f'{prefix}_queries.ids', len(rows))
        if rows.shape != teacher.shape or rows.ndim != 2 or rows.shape[1] != 100:
            raise ValueError('Expected aligned dense top-100 arrays')
        expected = (data / 'split' / f'{name}.qids').read_text().split()
        if set(ids) != set(expected) or len(ids) != len(expected):
            raise ValueError('Query IDs differ from the shared split')
        if limit > len(ids):
            raise ValueError(f'{name} query limit exceeds available queries')
        selected_rows = sorted(random.Random(seed).sample(range(len(ids)), limit))
        selected[name] = {ids[r]: r for r in selected_rows}
        for qid, row in selected[name].items():
            docs, values = rows[row].tolist(), teacher[row].tolist()
            if len(set(docs)) != len(docs) or min(docs) < 0 or not np.isfinite(values).all():
                raise ValueError('Invalid candidates or teacher scores')
            candidates[qid], scores[qid] = list(map(str, docs)), values
    if set(selected['train']) & set(selected['validation']):
        raise ValueError('Training and validation queries overlap')
    model, threshold = load_baseline(bundle / 'B0_final_atom.pt')
    torch.set_num_threads(4)
    queries = {}
    with open(bundle / 'latents_B0_final_atom/train.query.json', 'rb') as source:
        for qid, record in ijson.kvitems(source, '', use_float=True):
            if qid in selected['train']:
                queries[qid] = latent(record)
    if set(selected['train']) - queries.keys():
        raise ValueError('Training query activations are missing')
    val_embeddings = np.load(data / 'val_queries.npy', mmap_mode='r')
    with torch.no_grad():
        items = list(selected['validation'].items())
        for start in range(0, len(items), 64):
            chunk = items[start:start + 64]
            batch = torch.tensor(np.asarray(val_embeddings[[r for _, r in chunk]]), dtype=torch.float32)
            codes = model.threshold(model.pre_activation(batch), threshold)
            for (qid, _), code in zip(chunk, codes):
                ids = (code > 0).nonzero().flatten()
                queries[qid] = dict(zip(ids.tolist(), code[ids].tolist()))
    del model
    wanted = {docid for docs in candidates.values() for docid in docs}
    documents, df = {}, Counter()
    count = length = 0
    passage_hash = hashlib.sha256()
    # Scan the full frozen corpus for exact IDF/length statistics, but retain
    # only this pilot's candidate passages in RAM.
    with open(bundle / 'latents_B0_final_atom/passages.latent.jsonl', 'rb') as source:
        for line in source:
            passage_hash.update(line)
            record = json.loads(line)
            concepts = latent(record, max_concepts)
            docid = str(record['docid'])
            if docid != str(count):
                raise ValueError('Passage IDs must match sequential embedding row positions')
            if docid in wanted:
                if docid in documents:
                    raise ValueError('Duplicate candidate passage in latent corpus')
                documents[docid] = concepts
            df.update(concepts.keys())
            count += 1
            length += len(concepts)
            if count % 250000 == 0:
                print(f'Corpus statistics: {count:,} passages; {len(documents):,}/{len(wanted):,} candidate passages found', flush=True)
    if wanted - documents.keys():
        raise ValueError('Some dense candidates are absent from passage activations')
    if not count or not length:
        raise ValueError('Empty corpus')
    verification = bundle / 'verification.json'
    if verification.is_file():
        expected_count = json.loads(verification.read_text())['arrays']['data_titled/passages.npy']['shape'][0]
        if count != expected_count:
            raise ValueError('Passage activations do not cover the verified full corpus')
    output.mkdir(parents=True, exist_ok=False)
    stats = {'num_docs': count, 'avg_doc_length': length / count, 'length_definition': 'concept_count',
             'max_concepts': max_concepts, 'document_frequency': dict(df)}
    (output / 'stats.json').write_text(json.dumps(stats) + '\n')
    for name in ('train', 'validation'):
        with open(output / f'{name}.jsonl', 'w') as target:
            for qid in selected[name]:
                docs = [{'docid': d, 'concepts': documents[d], 'teacher_score': s}
                        for d, s in zip(candidates[qid], scores[qid])]
                target.write(json.dumps({'qid': qid, 'query': queries[qid], 'candidates': docs}) + '\n')
        (output / f'{name}.qids').write_text(''.join(q + '\n' for q in selected[name]))
    qrels = read_qrels(data / 'split/validation.qrels')
    (output / 'validation.qrels').write_text(''.join(
        f'{q} 0 {d} {r}\n' for q in selected['validation'] for d, r in qrels[q].items()))
    (output / 'pilot.json').write_text(json.dumps({
        'scope': 'Dense top-100 reranking pilot; not full-corpus retrieval or final evaluation',
        'bundle': str(bundle.resolve()), 'checkpoint_sha256': file_hash(bundle / 'B0_final_atom.pt'),
        'threshold': threshold, 'seed': seed, 'train_queries': train_limit,
        'validation_queries': validation_limit, 'corpus_passages': count, 'max_concepts': max_concepts,
        'input_sha256': {'passages.latent.jsonl': passage_hash.hexdigest(),
                         'train.query.json': file_hash(bundle / 'latents_B0_final_atom/train.query.json'),
                         'train.cands.npy': file_hash(data / 'train.cands.npy'),
                         'train.teacher.npy': file_hash(data / 'train.teacher.npy')}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--train-limit', type=int, default=256)
    parser.add_argument('--validation-limit', type=int, default=256)
    parser.add_argument('--max-concepts', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu', help='Device for the small scoring head, not SAE retraining')
    parser.add_argument('--reuse-prepared', action='store_true')
    args = parser.parse_args()
    root = Path(args.output_dir)
    if not args.reuse_prepared:
        prepare(args.bundle, root, args.train_limit, args.validation_limit, args.max_concepts, args.seed)
    elif not (root / 'pilot.json').is_file():
        parser.error('No prepared pilot to reuse')
    def run(module, options, filename):
        with open(root / filename, 'x') as log:
            subprocess.run([sys.executable, '-m', module, *map(str, options)], check=True, stdout=log)
    run('p1.train', ['--train', root / 'train.jsonl', '--validation', root / 'validation.jsonl',
        '--qrels', root / 'validation.qrels', '--stats', root / 'stats.json', '--output', root / 'model.pt',
        '--epochs', args.epochs, '--seed', args.seed, '--device', args.device], 'training.jsonl')
    common = ['--data', root / 'validation.jsonl', '--stats', root / 'stats.json', '--qrels', root / 'validation.qrels']
    for name, extra in [('released_bm25', []), ('tuned_bm25', ['--tune']),
                        ('learned', ['--checkpoint', root / 'model.pt'])]:
        run('p1.evaluate', common + extra + ['--run-output', root / f'{name}.run'], f'{name}.metrics.json')
    print(f'Pilot complete: {root}', flush=True)


if __name__ == '__main__':
    main()
