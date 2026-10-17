import unittest
from p1.data import bm25, features, sparse
from p1.metrics import evaluate


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.stats = {'num_docs': 10, 'avg_doc_length': 2., 'document_frequency': {1: 2, 2: 8}}

    def test_no_overlap(self):
        self.assertEqual(features({1: 1}, {2: 1}, self.stats), [])
        self.assertEqual(bm25({1: 1}, {2: 1}, self.stats), 0)

    def test_rare_concept_scores_higher(self):
        self.assertGreater(bm25({1: 1}, {1: 1}, self.stats), bm25({2: 1}, {2: 1}, self.stats))

    def test_bad_activations(self):
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                sparse({'1': value})

    def test_metrics_include_missing_queries(self):
        result = evaluate({'q1': ['bad', 'good']}, {'q1': {'good': 1}, 'q2': {'other': 1}})
        self.assertEqual(result['mrr_at_10'], .25)
        self.assertEqual(result['recall_at_1000'], .5)

    def test_ndcg_and_duplicates(self):
        self.assertEqual(evaluate({'q': ['a', 'b']}, {'q': {'a': 3, 'b': 1}})['ndcg_at_10'], 1.)
        with self.assertRaises(ValueError):
            evaluate({'q': ['a', 'a']}, {'q': {'a': 1}})


if __name__ == '__main__':
    unittest.main()
