"""Vocabulary-mismatch subsets from a BM25 run: python -m p5.mismatch --help

python -m p5.mismatch --run bm25.dev.trec --qrels qrels.dev.tsv --output-dir data/mismatch --name dev

A query is in the K subset when BM25 retrieves none of its relevant passages in the top K (paper Sec. 5.2.1,
App. F). Queries without a positive label are excluded and listed separately. Writes <name>.mismatch_<K>.qids
and <name>.mismatch.json with the counts.
"""
import argparse
import json
from pathlib import Path

from p1.metrics import read_qrels
from .harness import read_run

# Paper Table 5 / App. F counts on MS MARCO dev (6,980 queries); ours should be close for the dev subset.
PAPER_DEV = {10: 4300, 100: 2300, 1000: 988}


def mismatch_subsets(run, qrels, ks):
    subsets = {k: [] for k in ks}
    unlabeled = []
    for qid in sorted(qrels):
        relevant = {d for d, r in qrels[qid].items() if r > 0}
        if not relevant:
            unlabeled.append(qid)
            continue
        ranking = run.get(qid, [])
        for k in ks:
            if not relevant & set(ranking[:k]):
                subsets[k].append(qid)
    return subsets, unlabeled


def write_subsets(run_path, qrels_path, output_dir, ks, name):
    if any(k < 1 for k in ks):
        raise ValueError('Cutoffs must be positive')
    qrels = read_qrels(qrels_path)
    subsets, unlabeled = mismatch_subsets(read_run(run_path), qrels, ks)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    targets = [root / f'{name}.mismatch_{k}.qids' for k in ks] + [root / f'{name}.mismatch.json']
    existing = [str(t) for t in targets if t.exists()]
    if existing:
        raise FileExistsError(f'Choose a new output directory or name; {existing[0]} exists')
    for k, path in zip(ks, targets):
        path.write_text(''.join(q + '\n' for q in subsets[k]))
    summary = {'run': str(run_path), 'qrels': str(qrels_path),
               'counts': {str(k): len(subsets[k]) for k in ks},
               'labeled_queries': len(qrels) - len(unlabeled),
               'unlabeled_excluded': len(unlabeled), 'paper_dev_counts': {str(k): v for k, v in PAPER_DEV.items()}}
    targets[-1].write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True, help='BM25 run (TREC 6-column or 4-column)')
    parser.add_argument('--qrels', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--name', required=True, help='Split name used in file names, e.g. dev or validation')
    parser.add_argument('--ks', type=int, nargs='+', default=[10, 100, 1000])
    args = parser.parse_args()
    print(json.dumps(write_subsets(args.run, args.qrels, args.output_dir, args.ks, args.name), indent=2))


if __name__ == '__main__':
    main()
