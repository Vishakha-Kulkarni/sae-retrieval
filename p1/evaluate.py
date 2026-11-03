import argparse
import json
from .data import bm25, IndexedGroups, load_stats
from .metrics import evaluate, read_qrels


def main():
    p = argparse.ArgumentParser(description='Rerank fixed candidates and evaluate against TREC qrels')
    p.add_argument('--data', required=True)
    p.add_argument('--stats', required=True)
    p.add_argument('--qrels', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--k1', type=float, default=.6)
    p.add_argument('--b', type=float, default=2.)
    p.add_argument('--k2', type=float, default=1.5)
    p.add_argument('--tune', action='store_true', help='BM25 grid search; use validation data only')
    p.add_argument('--run-output', help='Write best ranking as TREC run')
    args = p.parse_args()
    if args.k1 <= 0 or args.b < 0 or args.k2 <= 0:
        p.error('Require k1 > 0, k2 > 0 and b >= 0')
    if args.tune and args.checkpoint:
        p.error('--tune only applies to BM25')
    groups, stats, qrels = IndexedGroups(args.data), load_stats(args.stats), read_qrels(args.qrels)
    if set(qrels) != groups.qids:
        raise ValueError('Candidate and qrels query IDs must match; supply the intended evaluation subset')
    if args.checkpoint:
        import torch
        from .model import ImpactScorer, score_group
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if checkpoint.get('format_version') != 2:
            raise ValueError('Retrain checkpoint: scoring features changed in format version 2')
        if checkpoint['stats'] != stats:
            raise ValueError('Corpus statistics differ from training')
        model = ImpactScorer(checkpoint['hidden'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
    best, best_run = None, None
    grid = [(k, b) for k in (.3, .6, 1.2, 2.4) for b in (0., .25, .5, .75, 1., 1.5, 2.)] if args.tune else [(args.k1, args.b)]
    for k1, b in grid:
        run, scored = {}, {}
        for group in groups:
            if args.checkpoint:
                with torch.no_grad():
                    scores = score_group(model, group, stats).tolist()
            else:
                scores = [bm25(group['query'], d['concepts'], stats, k1, b, args.k2) for d in group['candidates']]
            ranked = sorted(zip(scores, group['candidates']), key=lambda x: (-x[0], x[1]['docid']))
            run[group['qid']] = [d['docid'] for _, d in ranked]
            scored[group['qid']] = [(d['docid'], s) for s, d in ranked]
        result = {'method': 'learned' if args.checkpoint else 'released_bm25',
                  'k2': args.k2 if not args.checkpoint else None,
                  'k1': k1 if not args.checkpoint else None,
                  'b': b if not args.checkpoint else None, **evaluate(run, qrels)}
        if best is None or result['mrr_at_10'] > best['mrr_at_10']:
            best, best_run = result, scored
    print(json.dumps(best, indent=2))
    if args.run_output:
        with open(args.run_output, 'w') as output:
            for qid, ranking in best_run.items():
                for rank, (docid, score) in enumerate(ranking, 1):
                    output.write(f'{qid} Q0 {docid} {rank} {score:.9g} p1\n')


if __name__ == '__main__':
    main()
