"""P5 diagnostics and the report's worked example: python -m p5.example --help

python -m p5.example --queries val.query.json --expanded L.val.query.json --embeddings val_queries.npy \
    --checkpoint sae.pt --stats stats.json --expander expander.pt --database corpus.db \
    --baseline-run B0.val.run --run L.val.run --qrels validation.qrels --subset validation.mismatch_1000.qids \
    --query-text validation.queries.tsv --describe train.query.json --describe-text queries.train.tsv \
    --collection collection.tsv --output example.val.json

1. sources: where L's added concepts come from (near-miss pool, decoder neighbours, or both), by count and weight.
2. movement: how the rank of the first relevant passage changes from the baseline run to L (all queries and the
   subset), and, for queries the baseline misses at 1000, whether a relevant passage's indexed concepts include
   any concept L added.
3. example: the subset query whose first relevant passage gains most in reciprocal rank, its top added concepts
   described by the --describe queries that activate them most, and the relevant passage.
Read-only: every input is opened for reading and only --output is written.
"""
import argparse
import heapq
import json
from contextlib import closing
from pathlib import Path
import sqlite3

import numpy as np
import torch

from p1.data import load_stats
from p1.metrics import read_qrels
from p2.sae import load_sae
from .harness import read_qids, read_run
from .model import load_expander
from .pool import near_miss, pool_context
from .train import load_embeddings


def added_concepts(original, expanded):
    """{qid: [(concept, weight)]} that the expanded file appends to each original query."""
    out = {}
    for qid, query in expanded.items():
        base = original[qid]['ids']
        if query['ids'][:len(base)] != base:
            raise ValueError(f'{qid}: expanded query does not start with the original concepts')
        out[qid] = list(zip(query['ids'][len(base):], query['weight'][len(base):]))
    return out


def label(concept, near, neighbours):
    return {(True, True): 'both', (True, False): 'near_miss', (False, True): 'neighbour'}.get(
        (concept in near, concept in neighbours), 'other')


@torch.no_grad()
def sources(original, added, embeddings_path, checkpoint, stats_path, expander_path):
    """Per query {concept: source}, and totals by count and by added weight."""
    _, payload = load_expander(expander_path, checkpoint, stats_path)
    sae, _ = load_sae(checkpoint, payload['k'])
    sae.eval()
    _, table, _ = pool_context(sae, load_stats(stats_path), payload['pool']['per_concept'])
    matrix, row = load_embeddings(embeddings_path)
    per_query, count, weight = {}, {}, {}
    for qid, pairs in added.items():
        x = torch.as_tensor(np.asarray(matrix[row[qid]]), dtype=torch.float32)
        near = set(near_miss(sae.pre_activation(x[None]), payload['threshold'], payload['pool']['near_miss'])[0])
        neighbours = set(table[original[qid]['ids']].ravel().tolist()) if original[qid]['ids'] else set()
        per_query[qid] = {c: label(c, near, neighbours) for c, _ in pairs}
        for c, w in pairs:
            source = per_query[qid][c]
            count[source] = count.get(source, 0) + 1
            weight[source] = weight.get(source, 0.) + w
    return per_query, {'count': count, 'weight': weight}


def first_relevant(ranked, relevant):
    """1-based rank of the first relevant docid, or None if none is retrieved."""
    return next((rank for rank, docid in enumerate(ranked, 1) if docid in relevant), None)


def movement(baseline, run, qrels, qids):
    """Counts of how the first relevant rank changes, over qids."""
    counts = dict.fromkeys(('up', 'down', 'same', 'found', 'lost', 'missed_both'), 0)
    for qid in qids:
        relevant = {d for d, r in qrels[qid].items() if r > 0}
        before, after = first_relevant(baseline.get(qid, []), relevant), first_relevant(run.get(qid, []), relevant)
        if before is None and after is None:
            counts['missed_both'] += 1
        elif before is None:
            counts['found'] += 1
        elif after is None:
            counts['lost'] += 1
        else:
            counts['up' if after < before else 'down' if after > before else 'same'] += 1
    return counts


def passage_concepts(database, docids):
    uri = Path(database).resolve().as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as db:
        rows = {d: db.execute('SELECT concepts FROM documents WHERE docid=?', (d,)).fetchone() for d in docids}
    return {d: {int(c) for c in json.loads(r[0])} for d, r in rows.items() if r}


def miss_overlap(baseline, original, added, qrels, database):
    """For baseline misses at 1000: does a relevant passage hold any original / any added concept?"""
    missed = [q for q in qrels if first_relevant(baseline.get(q, []), {d for d, r in qrels[q].items() if r > 0})
              is None]
    concepts = passage_concepts(database, {d for q in missed for d, r in qrels[q].items() if r > 0})
    result = {'missed': len(missed), 'shares_original': 0, 'shares_added': 0, 'shares_neither': 0}
    for qid in missed:
        docs = [concepts.get(d, set()) for d, r in qrels[qid].items() if r > 0]
        has_original = any(set(original[qid]['ids']) & doc for doc in docs)
        has_added = any({c for c, _ in added.get(qid, [])} & doc for doc in docs)
        result['shares_original'] += has_original
        result['shares_added'] += has_added
        result['shares_neither'] += not (has_original or has_added)
    return result


def pick_example(baseline, run, qrels, qids):
    """(qid, rank before, rank after) with the largest reciprocal-rank gain; None if no query improves."""
    best = None
    for qid in sorted(qids):
        relevant = {d for d, r in qrels[qid].items() if r > 0}
        before, after = first_relevant(baseline.get(qid, []), relevant), first_relevant(run.get(qid, []), relevant)
        gain = (1 / after if after else 0.) - (1 / before if before else 0.)
        if gain > 0 and (best is None or gain > best[0]):
            best = gain, qid, before, after
    return best and best[1:]


def read_tsv(path, keys):
    """{first column: rest of the line} for the wanted keys, streamed."""
    found = {}
    with open(path) as source:
        for line in source:
            key, _, rest = line.rstrip('\n').partition('\t')
            if key in keys:
                found[key] = rest.replace('\t', ' | ')
                if len(found) == len(keys):
                    break
    return found


def describe(concepts, describe_path, text_path, per_concept=3):
    """{concept: [query text]} from the queries where each concept has the largest weight."""
    wanted = set(concepts)
    heaps = {c: [] for c in wanted}
    for qid, query in json.loads(Path(describe_path).read_text()).items():
        for c, w in zip(query['ids'], query['weight']):
            if c in wanted:
                heapq.heappush(heaps[c], (w, qid))
                if len(heaps[c]) > per_concept:
                    heapq.heappop(heaps[c])
    texts = read_tsv(text_path, {qid for heap in heaps.values() for _, qid in heap})
    return {c: [texts.get(q, q) for _, q in sorted(heap, reverse=True)] for c, heap in heaps.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, text in (('queries', 'B0 query latent JSON'), ('expanded', 'L query latent JSON'),
                       ('embeddings', 'Query .npy with .ids sibling'), ('checkpoint', 'SAE checkpoint'),
                       ('stats', 'stats.json of the index'), ('expander', 'p5.train checkpoint used for --expanded'),
                       ('database', 'p1.prepare corpus database of the index'), ('baseline-run', 'B0 run'),
                       ('run', 'L run'), ('qrels', 'Qrels'), ('subset', 'Mismatch qids for the example'),
                       ('query-text', 'qid<TAB>text for --queries'), ('describe', 'Query latent JSON for descriptions'),
                       ('describe-text', 'qid<TAB>text for --describe'), ('collection', 'docid<TAB>text'),
                       ('output', 'Result JSON')):
        parser.add_argument(f'--{name}', required=True, help=text)
    parser.add_argument('--top', type=int, default=5, help='Added concepts shown in the example')
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error(f'{args.output} exists; choose a new path')
    original = json.loads(Path(args.queries).read_text())
    added = added_concepts(original, json.loads(Path(args.expanded).read_text()))
    qrels, subset = read_qrels(args.qrels), read_qids(args.subset)
    baseline, run = read_run(args.baseline_run), read_run(args.run)
    per_query, totals = sources(original, added, args.embeddings, args.checkpoint, args.stats, args.expander)
    result = {'sources': totals,
              'movement': {'all': movement(baseline, run, qrels, sorted(qrels)),
                           'subset': movement(baseline, run, qrels, subset)},
              'baseline_misses': miss_overlap(baseline, original, added, qrels, args.database)}
    picked = pick_example(baseline, run, qrels, subset)
    if picked:
        qid, before, after = picked
        top = sorted(added[qid], key=lambda item: -item[1])[:args.top]
        descriptions = describe([c for c, _ in top], args.describe, args.describe_text)
        relevant = sorted(d for d, r in qrels[qid].items() if r > 0)
        passages = read_tsv(args.collection, set(relevant))
        indexed = passage_concepts(args.database, relevant)
        result['example'] = {
            'qid': qid, 'query': read_tsv(args.query_text, {qid}).get(qid), 'rank_before': before,
            'rank_after': after, 'original_concepts': len(original[qid]['ids']),
            'added': [{'concept': c, 'weight': w, 'source': per_query[qid][c], 'described_by': descriptions[c],
                       'in_relevant_passage': any(c in indexed.get(d, set()) for d in relevant)} for c, w in top],
            'relevant': [{'docid': d, 'text': passages.get(d, '')[:400]} for d in relevant]}
    with open(args.output, 'x') as target:
        json.dump(result, target, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
