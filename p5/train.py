"""Train the learned query expander (row L): python -m p5.train --help

python -m p5.train --train-groups train.groups.jsonl --train-embeddings train_queries.npy \
    --val-groups val.groups.jsonl --val-embeddings val_queries.npy --val-qrels validation.qrels \
    --checkpoint sae.pt --stats stats.json --output expander.pt

Groups come from p1.prepare groups: query latents from p2.evaluate export --queries with the same SAE,
the SimLM dense top-100 teacher run from p2.candidates, passages from p1.prepare corpus. The student score
is the released CL-SR BM25 (p5.bm25) of the expanded query, times a learned scale; the loss is
temperature-scaled KL to the SimLM teacher over the candidates plus L1 on the added weight. Every
--eval-every steps the top-B output for each --budgets value re-ranks the validation candidates (proxy
MRR@10, KTD8); the best (step, B) is saved with the SAE and stats hashes (KTD10).
"""
import argparse
import copy
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from p1.data import load_groups, load_stats
from p1.metrics import evaluate, read_qrels
from p2.sae import load_sae
from . import bm25
from .evaluate import file_hash
from .harness import restrict
from .model import FORMAT_VERSION, Expander, top_budget
from .pool import FEATURES, build_pool, decoder_directions, idf_vector, neighbour_table

MAX_MISMATCH = .01  # share of queries whose latents may differ from re-encoding (float ties at the threshold)


def load_embeddings(path):
    ids = Path(path).with_suffix('.ids').read_text().split()
    return np.load(path, mmap_mode='r'), {qid: i for i, qid in enumerate(ids)}


@torch.no_grad()
def prepare(groups, embeddings, sae, threshold, stats, near, per_concept):
    """Per query: pool concepts, their features, and the constant BM25 terms of the original and pool concepts."""
    matrix, row = embeddings
    directions = decoder_directions(sae)
    table, _ = neighbour_table(directions, per_concept)
    idf_values = idf_vector(stats, len(directions))
    examples, mismatched = [], 0
    for group in groups:
        if group['qid'] not in row:
            raise ValueError(f'{group["qid"]}: no query embedding')
        x = torch.as_tensor(np.asarray(matrix[row[group['qid']]]), dtype=torch.float32)
        pre = sae.pre_activation(x[None])[0]
        query = group['query']
        mismatched += set((pre > threshold).nonzero().flatten().tolist()) != set(query)
        concepts, features = build_pool(query, pre, x, directions, table, idf_values, threshold, near)
        docs = [d['concepts'] for d in group['candidates']]
        examples.append({
            'qid': group['qid'], 'docids': [d['docid'] for d in group['candidates']],
            'teacher': torch.tensor([float(d.get('teacher_score', 0.)) for d in group['candidates']]),
            'base': bm25.scores(bm25.doc_terms(docs, list(query), stats),
                                torch.tensor(list(query.values()), dtype=torch.float32), len(docs)),
            'terms': bm25.doc_terms(docs, concepts.tolist(), stats), 'features': features})
    if mismatched > MAX_MISMATCH * len(groups):
        raise ValueError(f'{mismatched}/{len(groups)} query latents differ from re-encoding the embeddings: '
                         'groups were exported with another SAE checkpoint or threshold')
    return examples


def student(model, example, budget=None):
    weights = model(example['features'])
    if budget is not None:
        weights = top_budget(weights, budget)
    return example['base'] + bm25.scores(example['terms'], weights, len(example['base'])), weights


@torch.no_grad()
def validate(model, examples, qrels, budget):
    run = {}
    for example in examples:
        scores = student(model, example, budget)[0].tolist()
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], example['docids'][i]))
        run[example['qid']] = [example['docids'][i] for i in order]
    return evaluate(run, qrels)


def train_expander(train, val, qrels, budgets=(4, 8, 16), steps=2000, batch=32, lr=1e-3, l1=1e-3,
                   temperature=1., hidden=32, eval_every=200, seed=42, log=print):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = Expander(len(FEATURES), hidden)
    model.set_normalisation(torch.cat([e['features'] for e in train]))
    log_scale = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([*model.parameters(), log_scale], lr=lr)
    model.eval()
    baseline = validate(model, val, qrels, 0)
    history = [{'step': 0, 'budget': 0, **baseline}]
    log(json.dumps(history[0]))
    best, window = None, []
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        loss = 0.
        for example in rng.sample(train, min(batch, len(train))):
            scores, weights = student(model, example)
            kl = F.kl_div(F.log_softmax(scores * log_scale.exp() / temperature, dim=0),
                          F.softmax(example['teacher'] / temperature, dim=0), reduction='sum') * temperature ** 2
            loss = loss + kl + l1 * weights.sum()
        loss = loss / min(batch, len(train))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss; inspect teacher and BM25 score scales')
        loss.backward()
        optimizer.step()
        window.append(loss.item())
        if step % eval_every == 0 or step == steps:
            model.eval()
            for budget in budgets:
                entry = {'step': step, 'budget': budget, 'train_loss': float(np.mean(window)),
                         'scale': log_scale.exp().item(), **validate(model, val, qrels, budget)}
                history.append(entry)
                log(json.dumps(entry))
                if best is None or entry['mrr_at_10'] > best[0]['mrr_at_10']:
                    best = entry, copy.deepcopy(model.state_dict())
            window = []
    return best, baseline, history


def train_from_files(train_groups, train_embeddings, val_groups, val_embeddings, val_qrels, checkpoint, stats_path,
                     output, threshold=None, k=32, near=32, per_concept=8, **options):
    if Path(output).exists():
        raise FileExistsError(f'{output} exists; choose a new path')
    train, val = load_groups(train_groups, require_teacher=True), load_groups(val_groups)
    overlap = {g['qid'] for g in train} & {g['qid'] for g in val}
    if overlap:
        raise ValueError(f'{len(overlap)} queries are in both train and validation, e.g. {sorted(overlap)[0]}')
    qrels = restrict(read_qrels(val_qrels), [g['qid'] for g in val])
    sae, meta = load_sae(checkpoint, k)
    sae.eval()
    threshold = threshold if threshold is not None else meta.get('threshold')
    if threshold is None:
        raise ValueError('Released checkpoints need --threshold (the mean_of_min_act in their filename)')
    stats = load_stats(stats_path)
    examples = [prepare(groups, load_embeddings(path), sae, threshold, stats, near, per_concept)
                for groups, path in ((train, train_embeddings), (val, val_embeddings))]
    (best, state), baseline, history = train_expander(*examples, qrels, **options)
    payload = {'format_version': FORMAT_VERSION, 'state_dict': state, 'features': FEATURES,
               'hidden': options.get('hidden', 32), 'budget': best['budget'], 'threshold': threshold, 'k': k,
               'pool': {'near_miss': near, 'per_concept': per_concept}, 'stats_max_concepts': stats.get('max_concepts'),
               'options': options, 'validation_metrics': best, 'baseline_validation_metrics': baseline, 'log': history,
               'input_sha256': {name: file_hash(path) for name, path in (
                   ('sae', checkpoint), ('stats', stats_path), ('train_groups', train_groups),
                   ('val_groups', val_groups), ('val_qrels', val_qrels))}}
    with open(output, 'xb') as target:
        torch.save(payload, target)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train-groups', required=True, help='p1.prepare groups with teacher scores')
    parser.add_argument('--train-embeddings', required=True, help='Train query .npy with .ids sibling')
    parser.add_argument('--val-groups', required=True)
    parser.add_argument('--val-embeddings', required=True)
    parser.add_argument('--val-qrels', required=True, help='qrels for the 5,000-query validation split only')
    parser.add_argument('--checkpoint', required=True, help='SAE checkpoint the query latents came from')
    parser.add_argument('--threshold', type=float, help='Defaults to the checkpoint mean_of_min_act')
    parser.add_argument('--k', type=int, default=32, help='Only for released upstream checkpoints')
    parser.add_argument('--stats', required=True, help='stats.json of the evaluated index')
    parser.add_argument('--output', required=True)
    parser.add_argument('--near', type=int, default=32, help='Near-miss concepts per query in the pool')
    parser.add_argument('--per-concept', type=int, default=8, help='Decoder neighbours per active concept in the pool')
    parser.add_argument('--budgets', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l1', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.budgets) < 1 or args.steps < 1 or args.batch < 1 or args.lr <= 0 or args.temperature <= 0:
        parser.error('budgets, steps, batch, lr and temperature must be positive')
    train_from_files(args.train_groups, args.train_embeddings, args.val_groups, args.val_embeddings, args.val_qrels,
                     args.checkpoint, args.stats, args.output, args.threshold, args.k, args.near, args.per_concept,
                     budgets=tuple(args.budgets), steps=args.steps, batch=args.batch, lr=args.lr, l1=args.l1,
                     temperature=args.temperature, hidden=args.hidden, eval_every=args.eval_every, seed=args.seed)


if __name__ == '__main__':
    main()
