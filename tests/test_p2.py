import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from p2.candidates import search
from p2.evaluate import encode, export, nmse
from p2.losses import listwise_loss, naacl_loss
from p2.sae import SAE, load_sae

ROOT = Path(__file__).resolve().parents[1]


def synthetic(root, n=600, d=16, queries=80, seed=0):
    """Clustered passages; each query is a noisy copy of one passage, which is its positive."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(20, d)) * 3
    passages = (centres[rng.integers(0, 20, n)] + rng.normal(size=(n, d))).astype(np.float32)
    np.save(root / 'passages.npy', passages)
    qrels = []
    for split, offset in (('train', 0), ('val', 10_000)):
        pos = rng.integers(0, n, queries)
        q = (passages[pos] + 0.3 * rng.normal(size=(queries, d))).astype(np.float32)
        qids = [str(offset + i) for i in range(queries)]
        np.save(root / f'{split}_queries.npy', q)
        (root / f'{split}_queries.ids').write_text(''.join(i + '\n' for i in qids))
        cands, scores = search(passages, q, top_k=20, chunk=128, device='cpu')
        np.save(root / f'{split}.cands.npy', cands)
        np.save(root / f'{split}.teacher.npy', scores)
        np.save(root / f'{split}.pos.npy', pos)
        if split == 'val':
            qrels = [f'{qid} 0 {p} 1\n' for qid, p in zip(qids, pos)]
    (root / 'val.qrels').write_text(''.join(qrels))
    return passages


class P2Tests(unittest.TestCase):
    def test_checkpoint_matches_upstream_parameter_names(self):
        model = SAE(16, 4, 3)
        self.assertEqual(set(model.upstream_state_dict()), {'linear.weight', 'linear2.weight', 'bias_pre'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'released.pt'
            torch.save(model.upstream_state_dict(), path)  # released checkpoints are bare state_dicts
            loaded, meta = load_sae(path, k=3)
            self.assertEqual(meta, {})
            self.assertTrue(torch.equal(loaded.linear.weight, model.linear.weight))

    def test_batch_topk_sparsity_and_decoder_norm(self):
        model = SAE(16, 4, 3)
        recon, z, _, _ = model(torch.randn(10, 16))
        self.assertLessEqual(int((z > 0).sum()), 30)
        self.assertEqual(recon.shape, (10, 16))
        self.assertTrue(torch.allclose(model.linear2.weight.norm(dim=-1), torch.ones(16), atol=1e-5))

    def test_atom_decoder_norm_and_checkpoint_round_trip(self):
        model = SAE(16, 4, 3, decoder_norm='atom')
        recon, _, _, _ = model(torch.randn(10, 16))
        recon.square().sum().backward()
        model.project_decoder_grad()
        self.assertTrue(torch.allclose(model.linear2.weight.norm(dim=0), torch.ones(64), atol=1e-5))
        # Projected gradient keeps each unit-norm decoder vector on its sphere (no radial component).
        radial = (model.linear2.weight.grad * model.linear2.weight).sum(0)
        self.assertTrue(torch.allclose(radial, torch.zeros(64), atol=1e-5))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'atom.pt'
            torch.save({'state_dict': model.upstream_state_dict(),
                        'meta': {'k': 3, 'args': {'decoder_norm': 'atom'}}}, path)
            self.assertEqual(load_sae(path)[0].decoder_norm, 'atom')

    def test_ranking_losses_vanish_when_student_equals_teacher(self):
        q, p = torch.randn(4, 8), torch.randn(4, 5, 8)
        self.assertAlmostEqual(listwise_loss(q, p, q, p).item(), 0., places=5)
        self.assertAlmostEqual(naacl_loss(q, p[:, 0], q, p[:, 0]).item(), 0., places=5)
        self.assertGreater(listwise_loss(q, p, q, -p).item(), 0.)
        # Standardised: a student that ranks identically at a different scale and offset costs nothing.
        self.assertAlmostEqual(listwise_loss(q, p, q * 0.01, p, standardized=True).item(), 0., places=5)
        self.assertGreater(listwise_loss(q, p, q * 0.01, p).item(), 0.01)

    def test_differentiable_bm25_matches_clsr_scorer(self):
        from scipy import sparse
        from p2.clsr import passage_impacts, query_matrix
        from p2.losses import bm25_scores
        rng = np.random.default_rng(0)
        zp = torch.tensor(rng.uniform(0, 3, (6, 40)) * (rng.random((6, 40)) < .3), dtype=torch.float32)
        zq = torch.tensor(rng.uniform(0, 3, (1, 40)) * (rng.random((1, 40)) < .4), dtype=torch.float32)
        impacts, df = passage_impacts(sparse.csr_matrix(zp.numpy()))
        n = zp.shape[0]
        idf = torch.tensor(np.log((n - df + 0.5) / (df + 0.5)), dtype=torch.float32)
        avgdl = float((zp > 0).sum()) / n
        queries = {'q': {'ids': [str(c) for c in np.flatnonzero(zq[0].numpy())],
                         'weight': zq[0][zq[0] > 0].tolist()}}
        _, qm = query_matrix(queries, 40, df)
        expected = (qm @ impacts.T).toarray()[0]
        ours = bm25_scores(zq, zp[None], idf, avgdl, torch.zeros(()))[0]
        np.testing.assert_allclose(ours.numpy(), expected, rtol=1e-4, atol=1e-5)

    def test_concept_comparison_bounds(self):
        from p2.concepts import compare
        z = torch.relu(torch.randn(300, 40)) * (torch.rand(300, 40) < .3)
        same = compare(z, z, top_n=10, min_fires=10)
        self.assertEqual(same['top_passage_jaccard_mean'], 1.0)
        self.assertEqual(same['passage_concepts_retained_mean'], 1.0)
        shuffled = compare(z, z[torch.randperm(300)], top_n=10, min_fires=10)
        self.assertLess(shuffled['top_passage_jaccard_mean'], 0.5)

    def test_chunked_search_is_exact(self):
        rng = np.random.default_rng(0)
        p, q = rng.normal(size=(300, 8)).astype(np.float32), rng.normal(size=(7, 8)).astype(np.float32)
        ids, scores = search(p, q, top_k=10, chunk=64, device='cpu')
        expected = np.argsort(-(q @ p.T), axis=1)[:, :10]
        np.testing.assert_array_equal(ids, expected)
        np.testing.assert_allclose(scores, np.take_along_axis(q @ p.T, expected, 1), rtol=1e-5)

    def test_export_formats_match_upstream(self):
        model = SAE(16, 4, 3)
        x = np.random.default_rng(0).normal(size=(5, 16)).astype(np.float32)
        recon, _ = encode(model, x, threshold=-1e9)
        self.assertAlmostEqual(nmse(x, torch.from_numpy(x.mean(0)).expand(5, 16)), 1., places=5)
        self.assertEqual(recon.shape, (5, 16))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export(model, 0., x, root / 'p.jsonl')
            export(model, 0., x, root / 'q.json', queries=True, ids=list('abcde'))
            first = json.loads((root / 'p.jsonl').read_text().splitlines()[0])
            self.assertEqual(set(first), {'docid', 'ids', 'weight'})
            self.assertEqual(first['weight'], sorted(first['weight'], reverse=True))
            queries = json.loads((root / 'q.json').read_text())
            self.assertEqual(set(queries), set('abcde'))
            self.assertTrue(all(isinstance(c, str) for q in queries.values() for c in q['ids']))

    def test_training_runs_end_to_end_for_every_objective(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic(root)
            base = [sys.executable, '-m', 'p2.train', '--data-dir', str(root), '--val-qrels', str(root / 'val.qrels'),
                    '--hidden-mult', '4', '--k', '4', '--batch', '128', '--steps', '40', '--eval-every', '20',
                    '--rank-batch', '16', '--rank-cands', '8', '--rank-head', '2', '--lr', '1e-3',
                    '--stats-sample', '600', '--nmse-sample', '600', '--device', 'cpu']
            runs = {'none': [], 'naacl': ['--rank-space', 'recon'],
                    'listwise': ['--rank-space', 'recon'], 'latent': ['--rank-space', 'latent']}
            for name, extra in runs.items():
                loss = 'listwise' if name == 'latent' else name
                out = root / name
                subprocess.run(base + ['--output', str(out), '--rank-loss', loss] + extra,
                               cwd=ROOT, check=True, capture_output=True, text=True)
                log = [json.loads(line) for line in (out / 'log.jsonl').read_text().splitlines()]
                self.assertEqual([r['step'] for r in log], [20, 40])
                self.assertTrue(np.isfinite(log[-1]['nmse']))
                self.assertTrue((out / 'best.pt').exists())
                self.assertTrue((out / 'step_20.pt').exists() and (out / 'step_40.pt').exists())
            # Fine-tuning from a checkpoint evaluates the starting point first (step 0).
            out = root / 'finetune'
            subprocess.run(base + ['--output', str(out), '--init', str(root / 'none/last.pt')],
                           cwd=ROOT, check=True, capture_output=True, text=True)
            steps = [json.loads(line)['step'] for line in (out / 'log.jsonl').read_text().splitlines()]
            self.assertEqual(steps, [0, 20, 40])
            # Ranking in differentiable CL-SR (BM25) space, fine-tuned from the checkpoint above.
            out_bm25 = root / 'finetune_bm25'
            subprocess.run(base + ['--output', str(out_bm25), '--init', str(out / 'last.pt'),
                                   '--rank-loss', 'listwise', '--rank-space', 'bm25'],
                           cwd=ROOT, check=True, capture_output=True, text=True)
            log = [json.loads(line) for line in (out_bm25 / 'log.jsonl').read_text().splitlines()]
            self.assertTrue(all(np.isfinite(r['nmse']) for r in log))
            metrics = subprocess.run([sys.executable, '-m', 'p2.evaluate', 'metrics', '--checkpoint',
                                      str(out / 'best.pt'), '--data-dir', str(root), '--qrels',
                                      str(root / 'val.qrels'), '--nmse-sample', '600', '--device', 'cpu'],
                                     cwd=ROOT, check=True, capture_output=True, text=True)
            self.assertIn('clsr_bm25', json.loads(metrics.stdout))
            paper = subprocess.run([sys.executable, '-m', 'p2.evaluate', 'table1', '--checkpoint',
                                    str(out / 'best.pt'), '--passages', str(root / 'passages.npy'),
                                    '--queries', str(root / 'val_queries.npy'), '--qrels',
                                    str(root / 'val.qrels'), '--device', 'cpu'],
                                   cwd=ROOT, check=True, capture_output=True, text=True)
            result = json.loads(paper.stdout)
            self.assertEqual(result['num_queries'], 80)
            self.assertGreater(result['recall_at_1000'], 0.99)  # top-1000 covers the 600-passage corpus


if __name__ == '__main__':
    unittest.main()
