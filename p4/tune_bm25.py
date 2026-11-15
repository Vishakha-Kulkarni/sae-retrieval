"""P4: tune the CL-SR BM25 settings (k1, b, k2) for one SAE on a validation split.

Setup (once): make the val qrels from P2's files
    python -m p4.tune_bm25 make-qrels --pos data/val.pos.npy --ids data/val_queries.ids --output data/val.qrels.txt

Run (unattended, resumable):
    python -m p4.tune_bm25 run --latents data/passages.latent.jsonl --queries data/val_b0.query.json \
        --qrels data/val.qrels.txt --workdir tune_b0

One setting at a time, keeping the best of each: k1 (rebuild per value), then b (rebuild), then k2
(search only). Every result is appended to <workdir>/results.jsonl, so a rerun skips finished
settings. Only the best index so far is kept on disk. Selection metric: MRR@10 on the val split.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


def make_qrels(pos_path, ids_path, output):
    pos = np.load(pos_path)
    ids = Path(ids_path).read_text().split()
    assert len(pos) == len(ids), f'{len(pos)} positives vs {len(ids)} query ids'
    lines = [f'{q} 0 {int(p)} 1' for q, p in zip(ids, pos) if p >= 0]
    Path(output).write_text('\n'.join(lines) + '\n')
    print(f'{len(lines)} val queries with a relevant passage -> {output}')


def patch_env_settings(p4_dir):
    """Let K1, B (indexing) and K2 (search) be set from environment variables. Idempotent."""
    edits = {'pyserini_indexing.py': ('K1, B = 0.6, 1.75',
                                      'K1, B = float(os.environ.get("K1", 0.6)), float(os.environ.get("B", 1.75))'),
             'pyserini_search.py': ('K2 = 2.5', 'K2 = float(os.environ.get("K2", 2.5))')}
    for name, (old, new) in edits.items():
        path = Path(p4_dir) / name
        text = path.read_text()
        if 'os.environ.get' in text:
            continue
        assert old in text, f'expected "{old}" in {path}'
        text = text.replace(old, new, 1)
        if '\nimport os\n' not in text:
            text = text.replace('\nimport json\n', '\nimport json\nimport os\n', 1)
        path.write_text(text)
        print(f'patched {path}: settings now read from environment')


def run(cmd, env):
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-3000:] + result.stderr[-3000:])
        raise SystemExit(f'failed: {" ".join(cmd)}')
    lines = result.stdout.splitlines()
    starts = [i for i, line in enumerate(lines) if line == '{']  # the report is the last top-level JSON
    return json.loads('\n'.join(lines[starts[-1]:])) if starts else {}


class Tuner:
    def __init__(self, args):
        self.args = args
        self.work = Path(args.workdir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.log = self.work / 'results.jsonl'
        self.done = {}
        if self.log.exists():
            for line in self.log.read_text().splitlines():
                r = json.loads(line)
                self.done[(r['k1'], r['b'], r['k2'])] = r
        self.best_index = None  # (k1, b, path) of the best index kept on disk

    def index_dir(self, k1, b):
        return self.work / f'index_k1_{k1}_b_{b}'

    def evaluate(self, k1, b, k2):
        key = (k1, b, k2)
        if key in self.done:
            r = self.done[key]
            print(f'  cached  k1={k1:<6} b={b:<5} k2={k2:<6} MRR@10 {r["MRR@10"]:.4f}', flush=True)
            return r['MRR@10']
        index = self.ensure_index(k1, b)
        env = dict(os.environ, K1=str(k1), B=str(b), K2=str(k2))
        report = run([sys.executable, '-m', 'p4.pyserini_search', '--index', str(index),
                      '--queries', self.args.queries, '--output', str(self.work / 'last.trec'),
                      '--qrels', self.args.qrels, '--repeats', '1'], env)
        m = report['metrics']
        record = {'k1': k1, 'b': b, 'k2': k2, 'MRR@10': m['MRR@10'], 'NDCG@10': m['NDCG@10'],
                  'Recall@1000': m['Recall@1000'], 'queries': m['evaluated_queries'],
                  'latency_ms': report['latency_ms']['mean']}
        with open(self.log, 'a') as f:
            f.write(json.dumps(record) + '\n')
        self.done[key] = record
        print(f'  result  k1={k1:<6} b={b:<5} k2={k2:<6} MRR@10 {m["MRR@10"]:.4f}  '
              f'R@1k {m["Recall@1000"]:.4f}', flush=True)
        return m['MRR@10']

    def prune_disk(self):
        """Keep only the index of the best (k1, b) found so far (each index is ~0.7 GB)."""
        best = max(self.done.values(), key=lambda r: r['MRR@10'])
        keep = self.index_dir(best['k1'], best['b'])
        for d in self.work.glob('index_k1_*'):
            if d != keep:
                shutil.rmtree(d, ignore_errors=True)

    def ensure_index(self, k1, b):
        index = self.index_dir(k1, b)
        if not (index / 'meta.json').exists():
            print(f'  build   k1={k1} b={b} ...', flush=True)
            run([sys.executable, '-m', 'p4.pyserini_indexing', '--latents', self.args.latents,
                 '--output', str(index), '--max-latents', str(self.args.max_latents),
                 '--threads', str(self.args.threads)], dict(os.environ, K1=str(k1), B=str(b)))
        return index

    def sweep(self, name, values, fixed):
        print(f'\n== {name} ==', flush=True)
        scores = {}
        for v in values:
            setting = dict(fixed, **{name: v})
            scores[v] = self.evaluate(setting['k1'], setting['b'], setting['k2'])
            self.prune_disk()
        best_v = max(scores, key=scores.get)
        print(f'best {name} = {best_v} (MRR@10 {scores[best_v]:.4f})', flush=True)
        return best_v


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    q = sub.add_parser('make-qrels', help='Build val qrels from P2 val.pos.npy + val_queries.ids')
    q.add_argument('--pos', required=True)
    q.add_argument('--ids', required=True)
    q.add_argument('--output', required=True)
    r = sub.add_parser('run', help='Tune k1, then b, then k2')
    r.add_argument('--latents', required=True, help='passage latents (.jsonl)')
    r.add_argument('--queries', required=True, help='VAL query latents (.json), same SAE')
    r.add_argument('--qrels', required=True, help='VAL qrels')
    r.add_argument('--workdir', required=True)
    r.add_argument('--max-latents', type=int, default=24)
    r.add_argument('--threads', type=int, default=8)
    r.add_argument('--k1s', default='0.6,3,6,14,30')
    r.add_argument('--bs', default='0.4,0.75,1.0,1.75')
    r.add_argument('--k2s', default='2.5,10,30,60,120')
    args = parser.parse_args()
    if args.command == 'make-qrels':
        make_qrels(args.pos, args.ids, args.output)
        return
    patch_env_settings(Path(__file__).parent)
    floats = lambda s: [float(x) for x in s.split(',')]
    tuner = Tuner(args)
    k1 = tuner.sweep('k1', floats(args.k1s), {'b': 1.75, 'k2': 2.5})
    b = tuner.sweep('b', floats(args.bs), {'k1': k1, 'k2': 2.5})
    k2 = tuner.sweep('k2', floats(args.k2s), {'k1': k1, 'b': b})
    best = tuner.done[(k1, b, k2)]
    tuner.ensure_index(k1, b)
    tuner.prune_disk()
    (tuner.work / 'best.json').write_text(json.dumps(best, indent=1))
    print('\nBEST on val:', json.dumps(best))
    print(f'Kept index: {tuner.index_dir(k1, b)}  (use K2={k2} when searching it)')


if __name__ == '__main__':
    main()