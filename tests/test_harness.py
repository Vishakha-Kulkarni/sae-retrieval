import json
from pathlib import Path
import tempfile
import unittest

from p1.metrics import evaluate, per_query, randomization_test
from p5.harness import compare, read_qids, read_run


class MetricTests(unittest.TestCase):
    def test_linear_gain_matches_trec_eval(self):
        qrels = {'q': {'a': 3, 'b': 1, 'c': 2}}
        run = {'q': ['b', 'a', 'x']}
        # trec_eval ndcg_cut: gain = grade; DCG 1 + 3/log2(3), ideal 3 + 2/log2(3) + 1/2.
        self.assertAlmostEqual(evaluate(run, qrels, gain='linear')['ndcg_at_10'], .607492, places=5)
        self.assertAlmostEqual(evaluate(run, qrels)['ndcg_at_10'], .576667, places=5)

    def test_binary_qrels_gain_independent(self):
        qrels = {'q': {'a': 1, 'b': 1}}
        run = {'q': ['x', 'b', 'y', 'a']}
        self.assertEqual(evaluate(run, qrels, gain='linear'), evaluate(run, qrels))

    def test_unknown_gain(self):
        with self.assertRaises(ValueError):
            evaluate({'q': ['a']}, {'q': {'a': 1}}, gain='log')

    def test_per_query_scores_missing_query_as_zero(self):
        scores = per_query({'q1': ['a']}, {'q1': {'a': 1}, 'q2': {'b': 1}})
        self.assertEqual(scores['q2'], {'mrr_at_10': 0., 'ndcg_at_10': 0., 'recall_at_1000': 0.})
        self.assertEqual(scores['q1']['mrr_at_10'], 1.)

    def test_per_query_skips_queries_without_positive_labels(self):
        self.assertEqual(set(per_query({}, {'q1': {'a': 1}, 'q2': {'a': 0}})), {'q1'})

    def test_duplicate_ranked_document(self):
        with self.assertRaises(ValueError):
            per_query({'q': ['a', 'a']}, {'q': {'a': 1}})


class RandomizationTests(unittest.TestCase):
    def test_identical_runs(self):
        scores = {f'q{i}': i / 10 for i in range(20)}
        self.assertEqual(randomization_test(scores, dict(scores)), 1.)

    def test_consistently_better_run(self):
        a = {f'q{i}': .5 for i in range(50)}
        b = {f'q{i}': .6 for i in range(50)}
        self.assertLess(randomization_test(a, b, permutations=10_000), .001)

    def test_seeded_and_symmetric(self):
        a = {f'q{i}': (i * 7 % 11) / 11 for i in range(40)}
        b = {f'q{i}': (i * 5 % 13) / 13 for i in range(40)}
        p = randomization_test(a, b, permutations=2000, seed=3)
        self.assertEqual(p, randomization_test(a, b, permutations=2000, seed=3))
        self.assertEqual(p, randomization_test(b, a, permutations=2000, seed=3))
        self.assertGreater(p, 0.)
        self.assertLessEqual(p, 1.)

    def test_query_sets_must_match(self):
        with self.assertRaises(ValueError):
            randomization_test({'q1': 1.}, {'q2': 1.})
        with self.assertRaises(ValueError):
            randomization_test({}, {})


class RunFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def write(self, name, text):
        path = self.root / name
        path.write_text(text)
        return path

    def test_four_and_six_column_runs_agree(self):
        upstream = self.write('a.run', 'q1\td2\t2\t1.0\nq1\td1\t1\t2.0\n')
        trec = self.write('b.run', 'q1 Q0 d1 1 2.0 x\nq1 Q0 d2 2 1.0 x\n')
        self.assertEqual(read_run(upstream), {'q1': ['d1', 'd2']})
        self.assertEqual(read_run(upstream), read_run(trec))

    def test_bad_runs(self):
        for text in ('q1\td1\t1\n', 'q1 d1 1 1.0\nq1 d1 2 0.5\n', 'q1 d1 0 1.0\n'):
            with self.assertRaises(ValueError):
                read_run(self.write('bad.run', text))

    def test_compare_with_subset(self):
        qrels = self.write('qrels', 'q1 0 a 1\nq2 0 b 1\nq3 0 c 1\n')
        run_a = self.write('a.run', 'q1 x 1 1\nq2 b 1 1\nq3 x 1 1\n')
        run_b = self.write('b.run', 'q1 a 1 1\nq2 b 1 1\nq3 c 1 1\n')
        subset = self.write('subset.qids', 'q1\nq3\n')
        result = compare(qrels, run_a, run_b, subset, permutations=500)
        self.assertEqual(result['a']['num_queries'], 2)
        self.assertEqual(result['a']['mrr_at_10'], 0.)
        self.assertEqual(result['b']['mrr_at_10'], 1.)
        self.assertIn('mrr_at_10', result['p_value'])
        json.dumps(result)

    def test_subset_query_outside_qrels(self):
        qrels = self.write('qrels', 'q1 0 a 1\n')
        run = self.write('a.run', 'q1 a 1 1\n')
        with self.assertRaises(ValueError):
            compare(qrels, run, run, self.write('subset.qids', 'q1\nq9\n'))

    def test_read_qids_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            read_qids(self.write('dup.qids', 'q1\nq1\n'))


if __name__ == '__main__':
    unittest.main()
