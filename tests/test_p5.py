import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import torch

from p1.data import bm25, load_stats
from p1.prepare import build_corpus
from p2.evaluate import export
from p2.sae import SAE, load_sae
from p5.bm25 import doc_terms, scores
from p5.evaluate import evaluate_row, retrieve
from p5.expand import expand_neighbours, expand_threshold, write_expanded, write_learned
from p5.harness import read_run
from p5.model import load_expander
from p5.pool import FEATURES, build_pool, decoder_directions, eligible, idf_vector, near_miss, neighbour_table
from p5.table import build_table, render_latex, render_markdown
from p5.train import train_from_files
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


def tiny_sae():
    """Four concepts over 4-d inputs: 0~1 and 2~3 are near-parallel pairs, 1 and 3 with non-unit norm."""
    model = SAE(input_dim=4, hidden_mult=1, k=2)
    columns = torch.tensor([[1., 0, 0, 0], [4.5, .5, 0, 0], [0, 0, 1, 0], [0, 0, 1.6, .4]])
    with torch.no_grad():
        model.linear2.weight.copy_(columns.T)
        model.linear.weight.copy_(torch.eye(4))
    return model


def write_stats(path, frequencies, num_docs=10):
    path.write_text(json.dumps({'num_docs': num_docs, 'avg_doc_length': 2., 'length_definition': 'concept_count',
                                'document_frequency': {str(c): f for c, f in frequencies.items()}}))
    return load_stats(path)


class PoolTests(unittest.TestCase):
    def test_neighbour_table_finds_hand_built_pairs(self):
        indices, cosines = neighbour_table(decoder_directions(tiny_sae()), 1, chunk=3)
        self.assertEqual(indices[:, 0].tolist(), [1, 0, 3, 2])
        self.assertTrue(((cosines >= -1) & (cosines <= 1 + 1e-6)).all())
        self.assertAlmostEqual(float(cosines[0, 0]), 4.5 / np.hypot(4.5, .5), places=5)

    def test_eligible_drops_out_of_vocabulary_and_nonpositive_idf(self):
        with tempfile.TemporaryDirectory() as directory:
            stats = write_stats(Path(directory) / 'stats.json', {0: 1, 1: 2, 3: 8})
        self.assertEqual(eligible(stats, 4).tolist(), [True, True, False, False])

    def test_near_miss_takes_strongest_concepts_below_threshold(self):
        pre = torch.tensor([[.9, .3, .5, -.2], [.1, .2, .05, .8]])
        self.assertEqual(near_miss(pre, .6, 2), [[2, 1], [1, 0]])


class ExpandTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.checkpoint = self.root / 'sae.pt'
        torch.save({'state_dict': tiny_sae().upstream_state_dict(), 'meta': {'k': 2, 'threshold': .3}},
                   self.checkpoint)
        self.stats = write_stats(self.root / 'stats.json', {0: 1, 1: 2, 2: 3, 3: 8})
        self.queries = {'q1': {'ids': [0], 'weight': [.7]}, 'q2': {'ids': [2, 1], 'weight': [.9, .123456789]}}

    def tearDown(self):
        self.directory.cleanup()

    def test_neighbours_keep_originals_and_add_only_eligible_new_concepts(self):
        expanded = expand_neighbours(self.queries, tiny_sae(), self.stats, budget=4, per_concept=3)
        for qid, query in self.queries.items():
            self.assertEqual(expanded[qid]['ids'][:len(query['ids'])], query['ids'])
            self.assertEqual(expanded[qid]['weight'][:len(query['weight'])], query['weight'])
            added = expanded[qid]['ids'][len(query['ids']):]
            self.assertFalse(set(added) & set(query['ids']))
            self.assertNotIn(3, added)  # IDF <= 0
        # q1: concept 1 is the near-parallel neighbour of 0; weight is source weight x cosine.
        self.assertEqual(expanded['q1']['ids'][1], 1)
        self.assertAlmostEqual(expanded['q1']['weight'][1], .7 * 4.5 / np.hypot(4.5, .5), places=5)
        self.assertEqual(expanded['q2']['ids'][2:], [0])

    def test_budget_zero_and_budget_cap(self):
        self.assertEqual(expand_neighbours(self.queries, tiny_sae(), self.stats, budget=0), self.queries)
        expanded = expand_neighbours(self.queries, tiny_sae(), self.stats, budget=1, per_concept=3)
        self.assertTrue(all(len(expanded[q]['ids']) - len(self.queries[q]['ids']) <= 1 for q in self.queries))

    def test_unknown_query_concept(self):
        with self.assertRaises(ValueError):
            expand_neighbours({'q': {'ids': [9], 'weight': [1.]}}, tiny_sae(), self.stats, budget=2)

    def test_threshold_scale_one_matches_export_and_lower_scale_is_superset(self):
        embeddings = self.root / 'q.npy'
        np.save(embeddings, np.array([[.9, .2, .4, .05], [.1, .5, .25, .7]], dtype=np.float32))
        (self.root / 'q.ids').write_text('a\nb\n')
        model, _ = load_sae(self.checkpoint)
        export(model, .3, np.load(embeddings), self.root / 'export.json', queries=True, ids=['a', 'b'])
        expand_threshold(self.checkpoint, embeddings, 1., self.root / 's1.json')
        self.assertEqual((self.root / 's1.json').read_text(), (self.root / 'export.json').read_text())
        expand_threshold(self.checkpoint, embeddings, .5, self.root / 's05.json')
        full, low = (json.loads((self.root / n).read_text()) for n in ('s1.json', 's05.json'))
        for qid in full:
            self.assertLessEqual(set(full[qid]['ids']), set(low[qid]['ids']))
        self.assertGreater(sum(map(len, (q['ids'] for q in low.values()))),
                           sum(map(len, (q['ids'] for q in full.values()))))
        with self.assertRaises(FileExistsError):
            expand_threshold(self.checkpoint, embeddings, .5, self.root / 's05.json')

    def test_expanded_file_runs_through_upstream_retrieval(self):
        root = self.root
        index_path = build_index(root, [{'docid': 0, 'ids': [0, 1], 'weight': [.5, .5]},
                                        {'docid': 1, 'ids': [2], 'weight': [.9]}] +
                                 [{'docid': i, 'ids': [3], 'weight': [.1]} for i in range(2, 10)])
        (root / 'q.json').write_text(json.dumps(self.queries))
        write_expanded(root / 'q.json', root / 'stats.json', self.checkpoint, root / 'n.json', budget=2)
        (root / 'qrels').write_text('q1 0 0 1\nq2 0 1 1\n')
        row = evaluate_row('N2', root / 'n.json', index_path, root / 'qrels')
        self.assertEqual(row['metrics']['all']['mrr_at_10'], 1.)
        self.assertGreater(row['avg_query_concepts'], 1.5)


def row_json(name, mrr, p=None, qrels='aa', baseline='b0run', **extra):
    row = {'row': name, 'input_sha256': {'qrels': qrels}, 'run_sha256': f'{name}run',
           'metrics': {'all': {'mrr_at_10': mrr, 'recall_at_1000': .9, 'ndcg_at_10': .5},
                       'mismatch_1000': {'mrr_at_10': mrr / 2}},
           'avg_query_concepts': 30., 'index_size_bytes': 1.27e9, 'latency_ms': {'mean': 150.}, **extra}
    if p is not None:
        row['baseline_run_sha256'] = baseline
        row['p_value'] = {'all': {'mrr_at_10': p, 'recall_at_1000': .5, 'ndcg_at_10': .5},
                          'mismatch_1000': {'mrr_at_10': p}}
    return row


class TableTests(unittest.TestCase):
    def rows(self):
        return [row_json('T0.7', .36, p=.2), row_json('B0', .35, run_sha256='b0run'), row_json('L', .38, p=.01)]

    def test_baseline_first_then_input_order_with_markers(self):
        table = build_table(self.rows())
        self.assertEqual([r['row'] for r in table], ['B0', 'T0.7', 'L'])
        markdown = render_markdown(table)
        lines = markdown.splitlines()
        self.assertIn('**0.380**†', lines[4])
        self.assertIn('0.360 |', lines[3])
        self.assertNotIn('†', lines[3])
        self.assertIn(r'\textbf{0.380}$^\dagger$', render_latex(table))

    def test_missing_metric_is_a_dash(self):
        table = build_table(self.rows())
        self.assertIn('| - |', render_markdown(table))  # no DL19 files given
        rows = self.rows()
        del rows[2]['metrics']['mismatch_1000']
        self.assertIn('| - |', render_markdown(build_table(rows)).splitlines()[4])

    def test_rows_on_different_qrels(self):
        rows = self.rows()
        rows[0]['input_sha256']['qrels'] = 'zz'
        with self.assertRaisesRegex(ValueError, 'qrels'):
            build_table(rows)

    def test_row_tested_against_another_baseline_run(self):
        rows = self.rows()
        rows[2]['baseline_run_sha256'] = 'old'
        with self.assertRaisesRegex(ValueError, 'baseline'):
            build_table(rows)
        rows = self.rows()  # hashes missing on both sides must not count as a match
        del rows[2]['baseline_run_sha256'], rows[1]['run_sha256']
        with self.assertRaisesRegex(ValueError, 'baseline'):
            build_table(rows)

    def test_test_collection_columns(self):
        dl19 = [row_json('B0', .0, qrels='dl', run_sha256='b0run', metrics={'all': {'ndcg_at_10': .6}}),
                row_json('L', .0, qrels='dl', metrics={'all': {'ndcg_at_10': .65}}, baseline_run_sha256='b0run',
                         p_value={'all': {'ndcg_at_10': .03}})]
        markdown = render_markdown(build_table(self.rows(), dl19=dl19))
        self.assertIn('**0.650**†', markdown.splitlines()[4])
        self.assertIn('0.600', markdown.splitlines()[2])


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        index_path = build_index(root)
        with open(index_path, 'rb') as source:
            self.index = pickle.load(source)
        build_corpus(root / 'passages.jsonl', root / 'corpus.db', root / 'stats.json')
        self.stats = load_stats(root / 'stats.json')
        self.docs = [{c: w for c, w in zip(p['ids'], p['weight'])} for p in PASSAGES]

    def tearDown(self):
        self.directory.cleanup()

    def test_matches_upstream_retrieval_scores(self):
        upstream = upstream_functions('latent_retrieve.py')
        query = {1: .6, 3: .8, 5: .3}
        official = dict(upstream['retrieve_one']([str(c) for c in query], list(query.values()),
                                                 self.index['inverted_index'], self.index['document_lengths'],
                                                 self.index['avgdl']))
        ours = scores(doc_terms(self.docs, list(query), self.stats), torch.tensor(list(query.values())),
                      len(self.docs))
        for docid, score in enumerate(ours.tolist()):
            self.assertAlmostEqual(score, official.get(docid, 0.), places=5)

    def test_zero_added_weight_is_the_original_query(self):
        query = {1: .6, 3: .8}
        columns = list(query) + [2, 4]
        ours = scores(doc_terms(self.docs, columns, self.stats), torch.tensor([.6, .8, 0., 0.]), len(self.docs))
        for doc, score in zip(self.docs, ours.tolist()):
            self.assertAlmostEqual(score, bm25({c: v for c, v in query.items() if c in doc}, doc, self.stats),
                                   places=5)

    def test_gradient_only_for_concepts_in_candidates(self):
        weights = torch.tensor([.5, .5, .5], requires_grad=True)
        scores(doc_terms(self.docs, [4, 2, 77], self.stats), weights, len(self.docs)).sum().backward()
        self.assertNotEqual(weights.grad[0].item(), 0.)
        self.assertNotEqual(weights.grad[1].item(), 0.)
        self.assertEqual(weights.grad[2].item(), 0.)


class LearnedExpanderTests(unittest.TestCase):
    """One query whose teacher-preferred passage only wins once near-miss concept 1 is added."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.checkpoint = self.root / 'sae.pt'
        torch.save({'state_dict': tiny_sae().upstream_state_dict(), 'meta': {'k': 2, 'threshold': .3}},
                   self.checkpoint)
        self.stats_path = self.root / 'stats.json'
        write_stats(self.stats_path, {0: 2, 1: 1, 2: 1, 3: 8})
        candidates = [{'docid': '0', 'concepts': {'0': .6}, 'teacher_score': 5.},
                      {'docid': '1', 'concepts': {'0': .4, '1': .6}, 'teacher_score': 10.},
                      {'docid': '2', 'concepts': {'2': .5}, 'teacher_score': 0.}]
        for split, qid in (('train', 't1'), ('val', 'v1')):
            (self.root / f'{split}.groups.jsonl').write_text(
                json.dumps({'qid': qid, 'query': {'0': .9}, 'candidates': candidates}) + '\n')
            np.save(self.root / f'{split}.npy', np.array([[.9, .25, 0, 0]], dtype=np.float32))
            (self.root / f'{split}.ids').write_text(qid + '\n')
        (self.root / 'val.qrels').write_text('v1 0 1 1\n')

    def tearDown(self):
        self.directory.cleanup()

    def train(self, output='expander.pt', **options):
        root = self.root
        return train_from_files(root / 'train.groups.jsonl', root / 'train.npy', root / 'val.groups.jsonl',
                                root / 'val.npy', root / 'val.qrels', self.checkpoint, self.stats_path,
                                root / output, **{'budgets': (1,), 'steps': 60, 'batch': 1, 'lr': .05,
                                                  'eval_every': 20, 'near': 4, 'per_concept': 3, **options})

    def test_pool_excludes_query_and_nonpositive_idf_concepts(self):
        model, stats = tiny_sae(), load_stats(self.stats_path)
        x = torch.tensor([.9, .25, 0, 0])
        directions = decoder_directions(model)
        table, _ = neighbour_table(directions, 3)
        concepts, features = build_pool({0: .9}, model.pre_activation(x[None])[0].detach(), x, directions, table,
                                        idf_vector(stats, 4), .3, 4)
        self.assertEqual(concepts.tolist(), [1, 2])
        self.assertEqual(features.shape, (2, len(FEATURES)))

    def test_training_lowers_loss_and_adds_the_helpful_concept(self):
        payload = self.train()
        self.assertEqual(payload['baseline_validation_metrics']['mrr_at_10'], .5)
        self.assertEqual(payload['validation_metrics']['mrr_at_10'], 1.)
        self.assertEqual(payload['budget'], 1)
        losses = [entry['train_loss'] for entry in payload['log'] if 'train_loss' in entry]
        self.assertLess(losses[-1], losses[0])
        model, _ = load_expander(self.root / 'expander.pt', self.checkpoint, self.stats_path)
        self.assertEqual(model.state_dict().keys(), payload['state_dict'].keys())

    def test_train_validation_overlap(self):
        (self.root / 'val.groups.jsonl').write_text((self.root / 'train.groups.jsonl').read_text())
        (self.root / 'val.ids').write_text('t1\n')
        (self.root / 'val.qrels').write_text('t1 0 1 1\n')
        with self.assertRaises(ValueError):
            self.train()

    def test_query_latents_from_another_sae(self):
        np.save(self.root / 'train.npy', np.array([[.1, .1, .9, 0]], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'SAE'):
            self.train()

    def test_learned_expansion_respects_budget_and_keeps_originals(self):
        self.train()
        queries = self.root / 'val.query.json'
        queries.write_text(json.dumps({'v1': {'ids': [0], 'weight': [.9]}}))
        args = queries, self.root / 'val.npy', self.checkpoint, self.stats_path, self.root / 'expander.pt'
        expanded = write_learned(*args, self.root / 'L.json')
        self.assertEqual(expanded['v1']['ids'], [0, 1])
        self.assertEqual(expanded['v1']['weight'][0], .9)
        self.assertEqual(json.loads((self.root / 'L.json').read_text()), expanded)
        self.assertEqual(write_learned(*args, self.root / 'L0.json', budget=0), json.loads(queries.read_text()))
        self.assertEqual(len(write_learned(*args, self.root / 'L5.json', budget=5)['v1']['ids']), 3)  # pool {1, 2}
        with self.assertRaises(FileExistsError):
            write_learned(*args, self.root / 'L.json')

    def test_learned_expansion_refuses_another_sae(self):
        self.train(steps=2, eval_every=1)
        other = self.root / 'other.pt'
        torch.save({'state_dict': SAE(4, 1, 2).upstream_state_dict(), 'meta': {'k': 2, 'threshold': .3}}, other)
        with self.assertRaisesRegex(ValueError, 'sae'):
            load_expander(self.root / 'expander.pt', other, self.stats_path)

    def test_checkpoint_refuses_other_statistics_and_existing_output(self):
        self.train(steps=2, eval_every=1)
        other = self.root / 'other.json'
        write_stats(other, {0: 2, 1: 1, 2: 2, 3: 8})
        with self.assertRaisesRegex(ValueError, 'stats'):
            load_expander(self.root / 'expander.pt', self.checkpoint, other)
        with self.assertRaises(FileExistsError):
            self.train(steps=2, eval_every=1)


if __name__ == '__main__':
    unittest.main()
