"""python -m p1.train --help"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from .data import load_groups, load_stats
from .metrics import evaluate, read_qrels
from .model import ImpactScorer, score_group


def main():
    parser = argparse.ArgumentParser(description='Distill dense candidate scores into additive concept impacts')
    parser.add_argument('--train', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--qrels', required=True, help='Validation qrels only')
    parser.add_argument('--stats', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.hidden < 1 or args.lr <= 0 or args.temperature <= 0:
        parser.error('epochs, hidden, lr and temperature must be positive')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = load_groups(args.train, require_teacher=True)
    validation = load_groups(args.validation)
    if {g['qid'] for g in train} & {g['qid'] for g in validation}:
        raise ValueError('Train and validation query IDs overlap')
    stats = load_stats(args.stats)
    qrels = read_qrels(args.qrels)
    if set(qrels) != {g['qid'] for g in validation}:
        raise ValueError('Validation qrels and validation candidate query IDs must match')
    model = ImpactScorer(args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = -1.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train)
        total = 0.
        for group in train:
            optimizer.zero_grad()
            scores = score_group(model, group, stats)
            teacher = torch.tensor([float(d['teacher_score']) for d in group['candidates']])
            temperature = args.temperature
            loss = F.kl_div(F.log_softmax(scores / temperature, dim=0),
                            F.softmax(teacher / temperature, dim=0), reduction='sum') * temperature ** 2
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss; inspect activation and teacher scales')
            loss.backward()
            optimizer.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            run = {}
            for group in validation:
                scores = score_group(model, group, stats).tolist()
                run[group['qid']] = [d['docid'] for _, d in sorted(
                    zip(scores, group['candidates']), key=lambda x: (-x[0], x[1]['docid']))]
            metrics = evaluate(run, qrels)
        print(json.dumps({'epoch': epoch + 1, 'loss': total / len(train), **metrics}))
        if metrics['mrr_at_10'] > best:
            best = metrics['mrr_at_10']
            torch.save({'format_version': 2, 'state_dict': model.state_dict(), 'hidden': args.hidden,
                        'stats': stats, 'args': vars(args), 'validation_metrics': metrics,
                        'input_sha256': {name: file_hash(path)
                                         for name, path in [('train', args.train), ('validation', args.validation),
                                                            ('qrels', args.qrels), ('stats', args.stats)]}}, args.output)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == '__main__':
    main()
