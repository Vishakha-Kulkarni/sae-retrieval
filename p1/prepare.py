"""CPU-only preparation: python -m p1.prepare --help."""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
import json
import gzip
import math
from pathlib import Path
import random
import sqlite3

from .data import sparse


def convert_collection(source_path, output_path, limit=None):
    """Convert the official collection TSV to the SimLM loader's JSONL.gz format."""
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    opener = gzip.open if str(source_path).endswith('.gz') else open
    with opener(source_path, 'rt') as source, gzip.open(output_path, 'xt') as target:
        for position, line in enumerate(source):
            if limit is not None and position >= limit:
                break
            docid, contents = line.rstrip('\n').split('\t', 1)
            # Official SimLM search resolves embedding row positions as passage IDs.
            if str(position) != docid:
                raise ValueError('SimLM requires sequential passage IDs matching row positions')
            target.write(json.dumps({'id': docid, 'title': '', 'contents': contents}) + '\n')


def latent(record, max_concepts=None):
    ids, weights = record['ids'], record['weight']
    if len(ids) != len(weights):
        raise ValueError('Concept IDs and weights have different lengths')
    if len({int(c) for c in ids}) != len(ids):
        raise ValueError('Duplicate concept IDs')
    # Upstream extraction writes active concepts. Reject zero entries rather
    # than silently changing the upstream concept-count length convention.
    values = sparse(dict(zip(ids, weights)))
    if len(values) != len(ids):
        raise ValueError('Latent exports must contain strictly positive active weights')
    pairs = sorted(values.items(), key=lambda item: -item[1])
    return dict(pairs if max_concepts is None else pairs[:max_concepts])


def build_corpus(passages, database, stats_path, max_concepts=None):
    if max_concepts is not None and max_concepts < 1:
        raise ValueError('max-concepts must be positive')
    if Path(database).exists() or Path(stats_path).exists():
        raise ValueError('Choose new output paths; corpus artifacts already exist')
    frequencies = Counter()
    count = length = 0
    with closing(sqlite3.connect(database)) as db, db:
        db.execute('CREATE TABLE documents (docid TEXT PRIMARY KEY, concepts TEXT NOT NULL)')
        with open(passages) as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                concepts = latent(record, max_concepts)
                db.execute('INSERT INTO documents VALUES (?, ?)',
                           (str(record['docid']), json.dumps(concepts)))
                count += 1
                length += len(concepts)
                frequencies.update(concepts.keys())
        if not count or not length:
            raise ValueError('Corpus must contain documents and active concepts')
    stats = {'num_docs': count, 'avg_doc_length': length / count,
             'length_definition': 'concept_count', 'max_concepts': max_concepts,
             'document_frequency': dict(frequencies)}
    Path(stats_path).write_text(json.dumps(stats) + '\n')
    return stats


def prepare_groups(queries_path, database, teacher_path, output, top_k=100, qids_path=None):
    if top_k < 1:
        raise ValueError('top-k must be positive')
    queries = json.loads(Path(queries_path).read_text())
    selected = set(Path(qids_path).read_text().split()) if qids_path else set(queries)
    if not selected or not selected <= queries.keys():
        raise ValueError('Selected query IDs must exist in query latents')
    candidates = defaultdict(dict)
    with open(teacher_path) as source:
        for line in source:
            if not line.strip():
                continue
            # Accept standard TREC runs or the official repo's qid/docid/rank/score output.
            fields = line.split()
            if len(fields) == 6:
                qid, _, docid, rank, score, _ = fields
            elif len(fields) == 4:
                qid, docid, rank, score = fields
            else:
                raise ValueError('Teacher run requires 4 or 6 columns')
            if qid not in selected:
                continue
            rank, score = int(rank), float(score)
            if rank < 1 or not math.isfinite(score):
                raise ValueError('Teacher ranks must be positive and scores finite')
            if docid in candidates[qid]:
                raise ValueError(f'{qid}: duplicate teacher document {docid}')
            candidates[qid][docid] = (rank, score)
    uri = Path(database).resolve().as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as db, open(output, 'x') as target:
        for qid in sorted(selected):
            entries = candidates[qid]
            ordered = sorted(entries, key=lambda docid: entries[docid][0])[:top_k]
            ranks = [entries[d][0] for d in ordered]
            if ranks != list(range(1, top_k + 1)):
                raise ValueError(f'{qid}: expected contiguous teacher ranks 1..{top_k}')
            docs = []
            for docid in ordered:
                row = db.execute('SELECT concepts FROM documents WHERE docid=?', (docid,)).fetchone()
                if row is None:
                    raise ValueError(f'Missing passage activations for {docid}')
                docs.append({'docid': docid, 'concepts': json.loads(row[0]),
                             'teacher_score': entries[docid][1]})
            target.write(json.dumps({'qid': qid, 'query': latent(queries[qid]), 'candidates': docs}) + '\n')


def split_queries(queries_path, qrels_path, output_dir, size=5000, seed=42):
    from .metrics import read_qrels
    queries = {}
    with open(queries_path) as source:
        for line in source:
            qid, text = line.rstrip('\n').split('\t', 1)
            if qid in queries:
                raise ValueError('Duplicate query ID')
            queries[qid] = text
    qrels = read_qrels(qrels_path)
    eligible = sorted(qid for qid, labels in qrels.items()
                      if qid in queries and any(r > 0 for r in labels.values()))
    if not 0 < size < len(eligible):
        raise ValueError('Validation size must leave at least one labeled training query')
    validation = set(random.Random(seed).sample(eligible, size))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    for split, ids in [('validation', validation), ('train', set(eligible) - validation)]:
        ids = sorted(ids)
        (root / f'{split}.qids').write_text(''.join(q + '\n' for q in ids))
        (root / f'{split}.queries.tsv').write_text(''.join(f'{q}\t{queries[q]}\n' for q in ids))
        (root / f'{split}.qrels').write_text(''.join(
            f'{q} 0 {d} {r}\n' for q in ids for d, r in qrels[q].items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    collection = commands.add_parser('collection', help='Convert downloaded MS MARCO collection.tsv[.gz] for SimLM')
    collection.add_argument('--input', required=True)
    collection.add_argument('--output', required=True, help='Output passages.jsonl.gz')
    collection.add_argument('--limit', type=int, help='First N passages for a local smoke subset; not final evaluation')
    corpus = commands.add_parser('corpus', help='Stream official passage latent JSONL into SQLite and corpus stats')
    corpus.add_argument('--passages', required=True)
    corpus.add_argument('--database', required=True)
    corpus.add_argument('--stats', required=True)
    corpus.add_argument('--max-concepts', type=int)
    groups = commands.add_parser('groups', help='Join official query latents, passage database and dense teacher run')
    groups.add_argument('--queries', required=True)
    groups.add_argument('--database', required=True)
    groups.add_argument('--teacher-run', required=True)
    groups.add_argument('--output', required=True)
    groups.add_argument('--top-k', type=int, default=100)
    groups.add_argument('--qids')
    split = commands.add_parser('split', help='Create deterministic train-only validation split; coordinate with P5')
    split.add_argument('--queries', required=True, help='MS MARCO training query TSV only')
    split.add_argument('--qrels', required=True, help='MS MARCO training qrels only')
    split.add_argument('--output-dir', required=True)
    split.add_argument('--size', type=int, default=5000)
    split.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.command == 'collection':
        convert_collection(args.input, args.output, args.limit)
    elif args.command == 'corpus':
        build_corpus(args.passages, args.database, args.stats, args.max_concepts)
    elif args.command == 'groups':
        prepare_groups(args.queries, args.database, args.teacher_run, args.output, args.top_k, args.qids)
    else:
        split_queries(args.queries, args.qrels, args.output_dir, args.size, args.seed)


if __name__ == '__main__':
    main()
