import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from p1.data import features, IndexedGroups, load_groups, load_stats
from p1.model import ImpactScorer, score_group
from p1.pilot import prepare
from p2.sae import SAE


class PilotTests(unittest.TestCase):
    def test_disk_index_handles_utf8_and_rejects_duplicate_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'groups.jsonl'
            groups = [{'qid': q, 'query': {'1': 1}, 'candidates': [
                {'docid': 'd', 'concepts': {'1': 1}, 'teacher_score': 1}]} for q in ('café', 'q2')]
            path.write_text('\n' + '\n'.join(json.dumps(g, ensure_ascii=False) for g in groups) + '\n')
            indexed = IndexedGroups(path, require_teacher=True)
            self.assertEqual(indexed.qids, {'café', 'q2'})
            self.assertEqual(indexed[1]['qid'], 'q2')
            self.assertEqual([g['qid'] for g in indexed], ['café', 'q2'])
            path.write_text(json.dumps(groups[0]) + '\n' + json.dumps(groups[0]) + '\n')
            with self.assertRaisesRegex(ValueError, 'duplicate qid'):
                IndexedGroups(path)

    def test_batched_scores_and_gradients_match_per_document(self):
        stats = {'num_docs': 10, 'avg_doc_length': 2., 'document_frequency': {1: 2, 2: 3}}
        group = {'query': {1: .8}, 'candidates': [
            {'concepts': {1: .9, 2: .2}}, {'concepts': {2: .5}}, {'concepts': {1: .2}}]}
        model = ImpactScorer()
        expected = torch.stack([model(torch.tensor(features(group['query'], d['concepts'], stats)).reshape(-1, 4))
                                for d in group['candidates']])
        actual = score_group(model, group, stats)
        self.assertTrue(torch.allclose(actual, expected))
        expected.sum().backward()
        grads = [p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        actual.sum().backward()
        self.assertTrue(all(torch.allclose(p.grad, g) for p, g in zip(model.parameters(), grads)))
        group['query'] = {}
        self.assertEqual(score_group(model, group, stats).tolist(), [0., 0., 0.])

    def test_prepare_preserves_split_and_full_corpus_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data_titled'
            (data / 'split').mkdir(parents=True)
            latents = root / 'latents_B0_final_atom'
            latents.mkdir()
            for name, prefix in [('train', 'train'), ('validation', 'val')]:
                (data / f'{prefix}_queries.ids').write_text(name + '\n')
                (data / f'split/{name}.qids').write_text(name + '\n')
                np.save(data / f'{prefix}.cands.npy', np.arange(100).reshape(1, 100))
                np.save(data / f'{prefix}.teacher.npy', np.arange(100, 0, -1, dtype=np.float32).reshape(1, 100))
            embeddings = np.zeros((1, 768), dtype=np.float32)
            embeddings[0, 0] = 1.
            np.save(data / 'val_queries.npy', embeddings)
            (data / 'split/validation.qrels').write_text('validation 0 0 1\n')
            (latents / 'train.query.json').write_text(json.dumps({'train': {'ids': [0], 'weight': [1.]}}))
            (latents / 'passages.latent.jsonl').write_text(''.join(json.dumps({
                'docid': d, 'ids': [d % 3], 'weight': [1.]}) + '\n' for d in range(120)))
            model = SAE(768, 1, k=1)
            with torch.no_grad():
                model.linear.weight.copy_(torch.eye(768))
            torch.save({'state_dict': model.state_dict(), 'meta': {'threshold': .25, 'k': 1}}, root / 'B0_final_atom.pt')
            out = root / 'pilot'
            prepare(root, out, 1, 1, 24, 42)
            stats = load_stats(out / 'stats.json')
            self.assertEqual(stats['num_docs'], 120)
            self.assertEqual(stats['document_frequency'], {0: 40, 1: 40, 2: 40})
            self.assertEqual(load_groups(out / 'train.jsonl')[0]['qid'], 'train')
            validation = load_groups(out / 'validation.jsonl')[0]
            self.assertEqual(validation['query'], {0: 1.})
            self.assertEqual(len(validation['candidates']), 100)


if __name__ == '__main__':
    unittest.main()
