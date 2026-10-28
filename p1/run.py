"""Run P1 from frozen latent exports and dense teacher runs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--passages', required=True, help='Frozen passage latent JSONL')
    for split in ('train', 'validation'):
        parser.add_argument(f'--{split}-queries', required=True, help='Query latent JSON dict')
        parser.add_argument(f'--{split}-run', required=True, help='Dense teacher TREC run')
        parser.add_argument(f'--{split}-qids', required=True, help='Agreed shared split IDs')
    parser.add_argument('--validation-qrels', required=True)
    parser.add_argument('--output-dir', required=True, help='New directory; existing outputs are never overwritten')
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-concepts', type=int, default=24,
                        help='Match agreed CL-SR Efficient index; default 24 retained passage concepts')
    args = parser.parse_args()
    for name, value in vars(args).items():
        if name not in ('output_dir', 'top_k', 'epochs', 'seed', 'max_concepts') and not Path(value).is_file():
            parser.error(f'Missing {name}: {value}')
    if args.top_k < 1 or args.epochs < 1 or args.max_concepts < 1:
        parser.error('top-k, epochs and max-concepts must be positive')
    train_ids = set(Path(args.train_qids).read_text().split())
    val_ids = set(Path(args.validation_qids).read_text().split())
    if not train_ids or not val_ids or train_ids & val_ids:
        parser.error('Shared train/validation IDs must be nonempty and disjoint')
    from .metrics import read_qrels
    if set(read_qrels(args.validation_qrels)) != val_ids:
        parser.error('Validation qrels must match shared validation IDs')
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'inputs.json').write_text(json.dumps(vars(args), indent=2) + '\n')

    def run(module, *options, log=None):
        command = [sys.executable, '-m', module, *map(str, options)]
        if log:
            with (root / log).open('w') as target:
                subprocess.run(command, check=True, stdout=target)
        else:
            subprocess.run(command, check=True)

    stats = root / 'stats.json'
    run('p1.prepare', 'corpus', '--passages', args.passages,
        '--database', root / 'corpus.db', '--stats', stats, '--max-concepts', args.max_concepts)
    for split in ('train', 'validation'):
        run('p1.prepare', 'groups', '--queries', getattr(args, f'{split}_queries'),
            '--database', root / 'corpus.db', '--teacher-run', getattr(args, f'{split}_run'),
            '--qids', getattr(args, f'{split}_qids'), '--top-k', args.top_k,
            '--output', root / f'{split}.jsonl')
    run('p1.train', '--train', root / 'train.jsonl', '--validation', root / 'validation.jsonl',
        '--stats', stats, '--qrels', args.validation_qrels, '--output', root / 'model.pt',
        '--epochs', args.epochs, '--seed', args.seed, log='training.jsonl')
    common = ['--data', root / 'validation.jsonl', '--stats', stats,
              '--qrels', args.validation_qrels]
    for method, extra in [('released_bm25', []), ('tuned_bm25', ['--tune']),
                          ('learned', ['--checkpoint', root / 'model.pt'])]:
        run('p1.evaluate', *common, *extra, '--run-output', root / f'{method}.run',
            log=f'{method}.metrics.json')
    print(f'P1 validation run complete: {root}')


if __name__ == '__main__':
    main()
