import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from p1.data import bm25
from p1.index import build, PostingIndex
from p1.metrics import evaluate, holm, randomization_test
from p1.model import ImpactScorer, score_group
from p1.retrieve import Retriever
from p1.final import validate, heldout
from p1.audit import audit
from p1.compare import compare
from p1.package import package
from p1.train import file_hash


class FullRetrievalTests(unittest.TestCase):
    def fixture(self, root):
        docs = [{0: 1.2, 1: .5}, {0: .8}, {2: 2.}, {0: .4, 2: .6}, {3: .7}]
        source = root / 'passages'
        source.write_text(''.join(json.dumps({'docid': i, 'ids': list(d), 'weight': list(d.values())}) + '\n'
                                  for i, d in enumerate(docs)))
        build(source, root / 'index', cap=2, batch_docs=2)
        return PostingIndex(root / 'index'), docs

    def test_posting_index_matches_formula_and_excludes_nonmatches(self):
        with tempfile.TemporaryDirectory() as directory:
            index, docs = self.fixture(Path(directory))
            scorer = Retriever(index, chunk=1)
            query = {0: .7}
            ranking, timing = scorer.search(query, 1000)
            self.assertEqual({d for d, _ in ranking}, {'0', '1', '3'})
            expected = {str(i): bm25(query, d, index.stats) for i, d in enumerate(docs) if 0 in d}
            for docid, score in ranking:
                self.assertAlmostEqual(score, expected[docid], places=6)
                self.assertLess(score, 0)
            self.assertEqual(timing['postings_touched'], 3)
            self.assertEqual(scorer.search({99: 1})[0], [])
            self.assertEqual(scorer.search({})[0], [])
            np.testing.assert_array_equal(index.postings(0)[0], [0, 1, 3])

    def test_neural_full_retrieval_matches_candidate_scores_and_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, docs = self.fixture(root)
            model = ImpactScorer()
            checkpoint = root / 'model.pt'
            def save():
                torch.save({'format_version': 2, 'hidden': 32, 'state_dict': model.state_dict(),
                            'stats': index.stats}, checkpoint)
            save()
            query = {0: .7, 1: .2}
            ranking, _ = Retriever(index, checkpoint, chunk=1).search(query)
            scores = score_group(model, {'query': query, 'candidates': [{'concepts': d} for d in docs]}, index.stats)
            for docid, score in ranking:
                self.assertAlmostEqual(score, scores[int(docid)].item(), places=5)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            save()
            self.assertEqual([d for d, _ in Retriever(index, checkpoint).search({0: 1}, 2)[0]], ['0', '1'])

    def test_linear_ndcg_and_paired_statistics(self):
        qrels = {'q': {'a': 3, 'b': 1}, 'missing': {'c': 0}}
        result = evaluate({'q': ['b', 'a']}, qrels, 'linear')
        expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3)) / 2
        self.assertAlmostEqual(result['ndcg_at_10'], expected)
        a, b = {str(i): 0. for i in range(20)}, {str(i): 1. for i in range(20)}
        self.assertEqual(randomization_test(a, a, 999), 1.)
        self.assertLess(randomization_test(a, b, 999), .01)
        self.assertEqual(holm([.01, .04]), [.02, .04])

    def test_protocol_selects_on_validation_and_rejects_changed_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'index').mkdir()
            (root / 'index/index.json').write_text(json.dumps({'source_sha256': 'frozen-index'}))
            (root / 'bundle/data_titled/split').mkdir(parents=True)
            val_qrels = root / 'bundle/data_titled/split/validation.qrels'
            val_qrels.write_text('v 0 1 1\n')
            models = {}
            for seed in ('42', '43', '44'):
                models[seed] = str(root / f'model-{seed}')
                Path(models[seed]).write_text(seed)
            calls = []
            def retrieve(index, queries, qrels, output, **kwargs):
                calls.append(Path(qrels))
                self.assertEqual(Path(qrels), val_qrels)
                return {'mrr_at_10': kwargs.get('k1', .6) / 10}
            with patch('p1.final.run_retrieval', retrieve):
                freeze = validate(root / 'index', root / 'queries', root / 'bundle', models, root / 'selection', 'cpu')
            self.assertEqual(len(calls), 34)
            self.assertEqual(freeze['selected_bm25']['k1'], 2.4)
            Path(models['42']).write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Checkpoint changed'):
                heldout(root / 'index', root / 'queries', root / 'bundle', root / 'selection/freeze.json',
                        root / 'train.ids', root / 'val.ids', root / 'test-results', 'cpu')
            self.assertFalse((root / 'test-results').exists())

    def test_text_audit_exposes_normalized_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'train.ids').write_text('t\n')
            (root / 'val.ids').write_text('v\n')
            (root / 'train.tsv').write_text('t\tWhat is SAE?\n')
            (root / 'val.tsv').write_text('v\twhat is sae\n')
            result = audit(root / 'train.ids', root / 'val.ids', root / 'train.tsv', root / 'val.tsv')
            self.assertEqual(result['overlapping_exact_query_texts'], 0)
            self.assertEqual(result['validation_ids_with_normalized_training_text'], ['v'])

    def test_significant_drop_is_not_reported_as_improvement(self):
        qrels = {str(i): {'good': 1} for i in range(20)}
        result = compare(qrels, {}, {'baseline': {q: ['good'] for q in qrels}}, permutations=999)
        row = result['paired_randomization_tests'][0]
        self.assertTrue(row['significant_at_05'])
        self.assertFalse(row['significant_improvement_at_05'])

    def test_report_waits_for_all_heldout_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / 'freeze.json'
            protocol.write_text('{}')
            summary = root / 'summary.json'
            summary.write_text(json.dumps({'protocol_sha256': file_hash(protocol), 'splits': {}}))
            with self.assertRaisesRegex(ValueError, 'all held-out splits'):
                package(summary, protocol, root / 'report')
            self.assertFalse((root / 'report').exists())

    def test_completed_report_uses_measured_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / 'freeze.json'
            protocol.write_text('{}')
            metric = {'mrr_at_10': .2, 'ndcg_at_10': .3, 'mean_latency_ms': 2.,
                      'index_bytes': 1024, 'gpu_name': 'Fixture GPU'}
            rows = {}
            for split in ('dev', 'dl19', 'dl20'):
                rows[split] = {'metrics': {name: metric for name in (
                    'released_bm25', 'tuned_bm25', 'learned_s42', 'learned_s43', 'learned_s44')},
                    'learned_three_seed_summary': {name: {'mean': .27, 'sample_std': .01}
                                                  for name in ('mrr_at_10', 'ndcg_at_10')},
                    'primary_seed_comparison': {'paired_randomization_tests': [
                        {'reference': 'released', 'p_holm': .1}, {'reference': 'tuned', 'p_holm': .2}]}}
            summary = root / 'summary.json'
            summary.write_text(json.dumps({'protocol_sha256': file_hash(protocol), 'splits': rows}))
            package(summary, protocol, root / 'report')
            self.assertIn('0.2700', (root / 'report/p1_contribution.tex').read_text())
            self.assertIn('Fixture GPU', (root / 'report/p1_contribution.tex').read_text())
            self.assertLess(len((root / 'report/p1_video_script.txt').read_text().split()), 180)


if __name__ == '__main__':
    unittest.main()
