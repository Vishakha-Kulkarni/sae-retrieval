import json
from pathlib import Path
import pickle
import tempfile
import unittest

from p5.evaluate import evaluate_row, retrieve
from p5.harness import read_run
from p5.upstream import upstream_functions


PASSAGES = [{'docid': 0, 'ids': [1, 2, 3], 'weight': [.7, .2, .1]},
            {'docid': 1, 'ids': [1, 3], 'weight': [.8, .5]},
            {'docid': 2, 'ids': [1], 'weight': [.4]},
            {'docid': 3, 'ids': [4, 5], 'weight': [.9, .3]},
            {'docid': 4, 'ids': [5], 'weight': [.6]}]


def build_index(root, passages=PASSAGES, max_latents=None):
    path = root / 'passages.jsonl'
    path.write_text(''.join(json.dumps(p) + '\n' for p in passages))
    upstream_functions('indexing.py')['build_inverted_index'](path, root / 'index.pkl', max_latents)
    return root / 'index.pkl'


class Fixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.index_path = build_index(self.root)
        with open(self.index_path, 'rb') as source:
            self.index = pickle.load(source)
        self.queries = {'10': {'ids': [3, 1], 'weight': [.8, .6]}, '11': {'ids': [5], 'weight': [.5]},
                        '12': {'ids': [99], 'weight': [.5]}}
        self.query_path = self.root / 'queries.json'
        self.query_path.write_text(json.dumps(self.queries))
        self.qrels_path = self.root / 'qrels'
        self.qrels_path.write_text('10 0 1 1\n11 0 4 1\n12 0 2 1\n')

    def tearDown(self):
        self.directory.cleanup()


class RowEvaluatorTests(Fixture):
    def test_run_matches_upstream_main(self):
        upstream = upstream_functions('latent_retrieve.py')
        for qid, query in self.queries.items():
            official = upstream['retrieve_one']([str(c) for c in query['ids']], query['weight'],
                                                self.index['inverted_index'], self.index['document_lengths'],
                                                self.index['avgdl'])
            ours, _ = retrieve(query, self.index, upstream)
            self.assertEqual(ours, official)

    def test_row_result(self):
        row = evaluate_row('B0', self.query_path, self.index_path, self.qrels_path,
                           run_output=self.root / 'b0.run')
        self.assertEqual(row['metrics']['all']['num_queries'], 3)
        self.assertEqual(row['metrics']['all']['mrr_at_10'], (1 + 1 + 0) / 3)
        self.assertEqual(row['avg_query_concepts'], 4 / 3)
        # Query 10 touches postings of concepts 3 (2 docs) and 1 (3 docs); 11 touches 5 (2 docs); 12 none.
        self.assertEqual(row['avg_postings_touched'], (5 + 2 + 0) / 3)
        self.assertEqual(row['index_size_bytes'], self.index_path.stat().st_size)
        self.assertEqual(set(row['latency_ms']), {'mean', 'p50', 'p95'})
        self.assertEqual(read_run(self.root / 'b0.run')['10'][0], '1')

    def test_out_of_vocabulary_query_scores_zero(self):
        row = evaluate_row('B0', self.query_path, self.index_path, self.qrels_path,
                           subsets={'oov': self.write('oov.qids', '12\n')})
        self.assertEqual(row['metrics']['oov']['mrr_at_10'], 0.)
        self.assertEqual(row['metrics']['oov']['num_queries'], 1)

    def test_baseline_p_values_per_subset(self):
        evaluate_row('B0', self.query_path, self.index_path, self.qrels_path, run_output=self.root / 'b0.run')
        row = evaluate_row('B0-again', self.query_path, self.index_path, self.qrels_path,
                           subsets={'s': self.write('s.qids', '10\n11\n')}, baseline_run=self.root / 'b0.run',
                           permutations=200)
        self.assertEqual(set(row['p_value']), {'all', 's'})
        self.assertEqual(row['p_value']['all']['mrr_at_10'], 1.)

    def test_qrels_query_missing_from_query_file(self):
        self.qrels_path.write_text('10 0 1 1\n77 0 1 1\n')
        with self.assertRaises(ValueError):
            evaluate_row('B0', self.query_path, self.index_path, self.qrels_path)

    def test_existing_outputs_are_not_overwritten(self):
        run = self.write('taken.run', '')
        with self.assertRaises(FileExistsError):
            evaluate_row('B0', self.query_path, self.index_path, self.qrels_path, run_output=run)

    def write(self, name, text):
        path = self.root / name
        path.write_text(text)
        return path


if __name__ == '__main__':
    unittest.main()
