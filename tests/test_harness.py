import json
from pathlib import Path
import tempfile
import unittest

from p1.metrics import evaluate, holm, per_query, randomization_test
from p5.harness import compare, filter_to_subset, primary_family, read_qids, read_run
from p5.mismatch import mismatch_subsets, write_subsets


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


class HolmTests(unittest.TestCase):
    def test_step_down_adjustment(self):
        # Sorted 0.012, 0.030, 0.041 against 0.05/3, 0.05/2, 0.05: only the smallest survives.
        adjusted = holm([.041, .012, .030])
        for value, expected in zip(adjusted, [.06, .036, .06]):
            self.assertAlmostEqual(value, expected)
        self.assertEqual([p < .05 for p in adjusted], [False, True, False])

    def test_adjusted_values_never_below_raw_and_capped(self):
        raw = [.001, .5, .9, .02]
        adjusted = holm(raw)
        self.assertTrue(all(a >= r for a, r in zip(adjusted, raw)))
        self.assertEqual(max(adjusted), 1.)
        self.assertEqual(holm([]), [])


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

    def test_primary_family(self):
        qrels = self.write('qrels', ''.join(f'q{i} 0 d{i} 1\n' for i in range(40)))
        subset = self.write('mismatch.qids', ''.join(f'q{i}\n' for i in range(20)))
        hit = self.write('hit.run', ''.join(f'q{i} d{i} 1 1\n' for i in range(40)))
        miss = self.write('miss.run', ''.join(f'q{i} x 1 1\n' for i in range(40)))
        result = primary_family(qrels, subset, ('L', hit), {'B0': miss, 'T': miss, 'N': hit}, permutations=2000)
        tests = result['tests']
        self.assertEqual(len(tests), 6)
        self.assertEqual({t['comparison'] for t in tests}, {'L vs B0', 'L vs T', 'L vs N'})
        self.assertEqual({(t['view'], t['metric']) for t in tests},
                         {('all', 'recall_at_1000'), ('mismatch_1000', 'mrr_at_10')})
        for t in tests:
            self.assertGreaterEqual(t['p_holm'], t['p'])
            self.assertEqual(t['significant'], t['comparison'] != 'L vs N')
        b0 = next(t for t in tests if t['comparison'] == 'L vs B0' and t['view'] == 'mismatch_1000')
        self.assertEqual((b0['ours'], b0['reference'], b0['num_queries']), (1., 0., 20))
        json.dumps(result)

    def test_read_qids_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            read_qids(self.write('dup.qids', 'q1\nq1\n'))


class SmokeFilterTests(unittest.TestCase):
    def test_keeps_only_queries_and_judgments_inside_the_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'qrels').write_text('q1 0 150 1\nq2 0 5 1\nq2 0 120 1\nq3 0 7 0\nq4 0 99 1\n')
            (root / 'queries.tsv').write_text('q1\tone\nq2\ttwo\nq3\tthree\nq4\tfour\nq5\tfive\n')
            kept = filter_to_subset(root / 'qrels', root / 'queries.tsv', 100, root / 'smoke')
            self.assertEqual(kept, ['q2', 'q4'])
            self.assertEqual((root / 'smoke.qrels').read_text(), 'q2 0 5 1\nq4 0 99 1\n')
            self.assertEqual((root / 'smoke.queries.tsv').read_text(), 'q2\ttwo\nq4\tfour\n')
            with self.assertRaises(FileExistsError):
                filter_to_subset(root / 'qrels', root / 'queries.tsv', 100, root / 'smoke')

    def test_query_text_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'qrels').write_text('q1 0 5 1\n')
            (root / 'queries.tsv').write_text('q9\tother\n')
            with self.assertRaises(ValueError):
                filter_to_subset(root / 'qrels', root / 'queries.tsv', 100, root / 'smoke')


class MismatchTests(unittest.TestCase):
    def setUp(self):
        self.qrels = {'rank50': {'g': 1}, 'two_rel': {'g1': 1, 'g2': 1}, 'no_hits': {'g': 1},
                      'deep': {'g': 1}, 'unlabeled': {'g': 0}}
        filler = [f'x{i}' for i in range(1000)]
        self.run = {'rank50': filler[:49] + ['g'] + filler[49:999],
                    'two_rel': filler[:4] + ['g1'] + filler[4:500] + ['g2'],
                    'deep': filler[:999] + ['g']}

    def test_failures_per_cutoff(self):
        subsets, unlabeled = mismatch_subsets(self.run, self.qrels, (10, 100, 1000))
        self.assertEqual(subsets[10], ['deep', 'no_hits', 'rank50'])
        self.assertEqual(subsets[100], ['deep', 'no_hits'])
        self.assertEqual(subsets[1000], ['no_hits'])
        self.assertEqual(unlabeled, ['unlabeled'])

    def test_subsets_nest(self):
        subsets, _ = mismatch_subsets(self.run, self.qrels, (1000, 10, 100))
        self.assertLessEqual(set(subsets[1000]), set(subsets[100]))
        self.assertLessEqual(set(subsets[100]), set(subsets[10]))

    def test_write_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'qrels').write_text(''.join(f'{q} 0 {d} {r}\n' for q, j in self.qrels.items()
                                                for d, r in j.items()))
            (root / 'bm25.run').write_text(''.join(f'{q} Q0 {d} {i} {-i} bm25\n' for q, docs in self.run.items()
                                                   for i, d in enumerate(docs, 1)))
            summary = write_subsets(root / 'bm25.run', root / 'qrels', root / 'out', (10, 1000), 'dev')
            self.assertEqual(summary['counts'], {'10': 3, '1000': 1})
            self.assertEqual(read_qids(root / 'out' / 'dev.mismatch_1000.qids'), ['no_hits'])
            with self.assertRaises(FileExistsError):
                write_subsets(root / 'bm25.run', root / 'qrels', root / 'out', (10,), 'dev')


if __name__ == '__main__':
    unittest.main()
