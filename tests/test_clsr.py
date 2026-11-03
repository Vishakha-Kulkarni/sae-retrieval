from collections import defaultdict
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tests.test_upstream import official_functions

ROOT = Path(__file__).resolve().parents[1]


class FastCLSRParity(unittest.TestCase):
    def test_scores_match_upstream_indexing_and_retrieval(self):
        rng = np.random.default_rng(0)
        indexing = official_functions('indexing.py', {
            'json': json, 'math': math, 'pickle': pickle, 'defaultdict': defaultdict,
            'tqdm': lambda values, **kw: values})
        scoring = official_functions('latent_retrieve.py', {
            'k1': .6, 'k2': 1.5, 'b': 2., 'TOP_K': 1000, 'defaultdict': defaultdict})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with open(root / 'passages.jsonl', 'w') as out:
                for d in range(300):
                    ids = rng.choice(60, rng.integers(1, 30), replace=False)
                    out.write(json.dumps({'docid': d, 'ids': ids.tolist(),
                                          'weight': rng.uniform(.1, 5, len(ids)).round(4).tolist()}) + '\n')
            # String concept IDs, as p2.evaluate export writes them; 99 never occurs in any passage.
            queries = {str(q): {'ids': [str(c) for c in rng.choice(60, 8, replace=False)] + ['99'],
                                'weight': rng.uniform(.1, 5, 9).round(4).tolist()} for q in range(20)}
            (root / 'q.json').write_text(json.dumps(queries))
            indexing['build_inverted_index'](root / 'passages.jsonl', root / 'index.pkl', 12)
            index = pickle.load(open(root / 'index.pkl', 'rb'))
            with open(root / 'upstream.tsv', 'w') as out:
                for qid, q in queries.items():
                    ranked = scoring['retrieve_one'](q['ids'], q['weight'], index['inverted_index'],
                                                     index['document_lengths'], index['avgdl'])
                    for rank, (d, s) in enumerate(ranked, 1):
                        out.write(f'{qid}\t{d}\t{rank}\t{s:.3f}\n')
            result = subprocess.run([sys.executable, '-m', 'p2.clsr', '--passage-latents', str(root / 'passages.jsonl'),
                                     '--query-latents', str(root / 'q.json'), '--output', str(root / 'ours.tsv'),
                                     '--max-latents', '12', '--check-run', str(root / 'upstream.tsv')],
                                    cwd=ROOT, check=True, capture_output=True, text=True)
            parity = json.loads(result.stdout)['parity_with_upstream']
            self.assertEqual(parity['queries_compared'], 19)  # last query is held out as possibly partial
            self.assertEqual(parity['mismatched'], 0)
            self.assertEqual(parity['same_top10_set'], 19)
            ours = (root / 'ours.tsv').read_text().splitlines()
            self.assertEqual(len(ours), len((root / 'upstream.tsv').read_text().splitlines()))


if __name__ == '__main__':
    unittest.main()
