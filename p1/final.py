"""Separate validation selection from locked held-out full-corpus evaluation."""
import argparse
import json
from pathlib import Path
import statistics
import shutil
import subprocess
import sys

from .compare import compare, read_run
from .metrics import read_qrels
from .train import file_hash


def run_retrieval(index, queries, qrels, output, checkpoint=None, k1=.6, b=2., limit=None, device='cuda'):
    command = [sys.executable, '-u', '-m', 'p1.retrieve', '--index', str(index), '--queries', str(queries),
               '--qrels', str(qrels), '--output-prefix', str(output), '--device', device,
               '--k1', str(k1), '--b', str(b), '--gain', 'linear']
    if checkpoint:
        command += ['--checkpoint', str(checkpoint)]
    if limit:
        command += ['--limit', str(limit), '--sample-seed', '42']
    subprocess.run(command, check=True)
    return json.loads(Path(str(output) + '.metrics.json').read_text())


def validate(index, query_dir, bundle, models, output, device):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    query = Path(query_dir) / 'validation.query.json'
    qrels = Path(bundle) / 'data_titled/split/validation.qrels'
    trials = []
    # Fixed, predeclared validation-only grid. No development/TREC feedback.
    grid = [(k, b) for k in (.3, .6, 1.2, 2.4) for b in (0., .25, .5, .75, 1., 1.5, 2.)]
    for i, (k, b) in enumerate(grid):
        result = run_retrieval(index, query, qrels, root / f'screen-{i}', k1=k, b=b, limit=500, device=device)
        trials.append({'k1': k, 'b': b, 'mrr_at_10': result['mrr_at_10']})
    shortlist = [(r['k1'], r['b']) for r in sorted(trials, key=lambda r: -r['mrr_at_10'])[:3]]
    # Always confirm the released setting and the setting selected by the earlier
    # candidate-only experiment, so the tuned full-index baseline cannot be weaker.
    shortlist = list(dict.fromkeys(shortlist + [(.6, 2.), (1.2, 1.)]))
    confirmations = []
    for i, (k, b) in enumerate(shortlist):
        result = run_retrieval(index, query, qrels, root / f'confirm-{i}', k1=k, b=b, device=device)
        confirmations.append({'k1': k, 'b': b, 'prefix': str(root / f'confirm-{i}'), 'metrics': result})
    best = max(confirmations, key=lambda r: r['metrics']['mrr_at_10'])
    released = next(r for r in confirmations if r['k1'] == .6 and r['b'] == 2.)
    metrics = {}
    for name, record in [('released_bm25', released), ('tuned_bm25', best)]:
        metrics[name] = record['metrics']
        for extension in ('.run', '.metrics.json'):
            source = Path(record['prefix'] + extension)
            if source.exists():
                shutil.copyfile(source, Path(str(root / name) + extension))
    metrics['learned_s42'] = run_retrieval(index, query, qrels, root / 'learned_s42', checkpoint=models['42'], device=device)
    freeze = {'format_version': 1, 'primary_seed': '42',
              'selection_scope': '28-setting BM25 grid on 500 validation queries, then top 3 plus released and prior candidate-tuned settings confirmed on all 5000; neural checkpoints selected on validation only',
              'trials': trials, 'full_validation_confirmation': confirmations,
              'selected_bm25': {'k1': best['k1'], 'b': best['b'], 'k2': 1.5},
              'models': {seed: {'path': str(Path(path).resolve()), 'sha256': file_hash(path)} for seed, path in models.items()},
              'index_source_sha256': json.loads((Path(index) / 'index.json').read_text())['source_sha256'],
              'validation_qrels_sha256': file_hash(qrels), 'validation_metrics': metrics,
              'heldout_rule': 'No architecture, epoch, seed, sparsity, or BM25 selection using held-out results'}
    (root / 'freeze.json').write_text(json.dumps(freeze, indent=2) + '\n')
    return freeze


def heldout(index, query_dir, bundle, freeze_path, train_ids, validation_ids, output, device):
    freeze = json.loads(Path(freeze_path).read_text())
    if freeze['format_version'] != 1 or set(freeze['models']) != {'42', '43', '44'}:
        raise ValueError('Need a locked protocol with all three seeds')
    if json.loads((Path(index) / 'index.json').read_text())['source_sha256'] != freeze['index_source_sha256']:
        raise ValueError('Index changed after selection')
    for record in freeze['models'].values():
        if file_hash(record['path']) != record['sha256']:
            raise ValueError('Checkpoint changed after selection')
    forbidden = set(Path(train_ids).read_text().split()) | set(Path(validation_ids).read_text().split())
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    data = Path(bundle) / 'data_titled'
    splits = [('dev', data / 'qrels.dev.small.tsv'), ('dl19', data / 'dl19.qrels'), ('dl20', data / 'dl20.qrels')]
    summary = {'scope': 'Locked full-corpus held-out evaluation', 'protocol_sha256': file_hash(freeze_path), 'splits': {}}
    tuned = freeze['selected_bm25']
    for split, qrels in splits:
        labels = read_qrels(qrels)
        if forbidden & labels.keys():
            raise ValueError(f'{split}: held-out query ID overlaps training or validation')
        queries = Path(query_dir) / f'{split}.query.json'
        current = {}
        current['released_bm25'] = run_retrieval(index, queries, qrels, root / f'{split}-released', device=device)
        current['tuned_bm25'] = run_retrieval(index, queries, qrels, root / f'{split}-tuned',
                                            k1=tuned['k1'], b=tuned['b'], device=device)
        for seed, record in freeze['models'].items():
            current['learned_s' + seed] = run_retrieval(index, queries, qrels, root / f'{split}-s{seed}',
                                                       checkpoint=record['path'], device=device)
        comparisons = compare(labels, read_run(root / f'{split}-s42.run'),
                              {'released': read_run(root / f'{split}-released.run'),
                               'tuned': read_run(root / f'{split}-tuned.run')},
                              gain='linear', metric='mrr_at_10' if split == 'dev' else 'ndcg_at_10')
        means = {}
        for metric in ('mrr_at_10', 'ndcg_at_10', 'recall_at_1000'):
            values = [current['learned_s' + seed][metric] for seed in freeze['models']]
            means[metric] = {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values), 'values': values}
        summary['splits'][split] = {'metrics': current, 'learned_three_seed_summary': means,
                                    'primary_seed_comparison': comparisons}
        (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['validation', 'heldout'])
    for name in ('index', 'query-dir', 'bundle', 'output-dir'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--models', help='JSON mapping 42/43/44 to fixed checkpoint paths')
    p.add_argument('--freeze')
    p.add_argument('--train-ids')
    p.add_argument('--validation-ids')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.phase == 'validation':
        if not a.models:
            p.error('validation requires --models')
        models = json.loads(Path(a.models).read_text())
        if set(models) != {'42', '43', '44'}:
            p.error('models must include fixed seeds 42, 43 and 44')
        validate(a.index, a.query_dir, a.bundle, models, a.output_dir, a.device)
    else:
        if not all((a.freeze, a.train_ids, a.validation_ids)):
            p.error('heldout requires --freeze, --train-ids and --validation-ids')
        heldout(a.index, a.query_dir, a.bundle, a.freeze, a.train_ids, a.validation_ids, a.output_dir, a.device)


if __name__ == '__main__':
    main()
