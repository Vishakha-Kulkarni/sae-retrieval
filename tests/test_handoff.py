import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import numpy as np
import torch

from p1.handoff import load_baseline, validate_bundle
from p2.sae import SAE


class HandoffTest(unittest.TestCase):
    def bundle(self, root):
        np.save(root / 'passages.npy', np.zeros((3, 768), dtype=np.float32))
        for name, prefix in [('train', 'train'), ('validation', 'val')]:
            np.save(root / f'{prefix}_queries.npy', np.zeros((1, 768), dtype=np.float32))
            (root / f'{prefix}_queries.ids').write_text(name + '\n')
            (root / f'{name}.qids').write_text(name + '\n')
            (root / f'{name}.qrels').write_text(f'{name} 0 0 1\n')
            np.save(root / f'{prefix}.cands.npy', np.array([[0, 1]], dtype=np.int64))
            np.save(root / f'{prefix}.teacher.npy', np.array([[2., 1.]], dtype=np.float32))

    def test_alignment_and_invalid_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.bundle(root)
            self.assertEqual(validate_bundle(root, root)['splits']['validation']['queries'], 1)
            (root / 'train_queries.ids').write_text('train\nextra\n')
            with self.assertRaisesRegex(ValueError, 'unique query IDs'):
                validate_bundle(root, root)
            (root / 'train_queries.ids').write_text('train\n')
            np.save(root / 'train.cands.npy', np.array([[0, 9]], dtype=np.int64))
            with self.assertRaisesRegex(ValueError, 'out of bounds'):
                validate_bundle(root, root)
            np.save(root / 'train.cands.npy', np.array([[0, 0]], dtype=np.int64))
            with self.assertRaisesRegex(ValueError, 'duplicate candidates'):
                validate_bundle(root, root)

    def test_checkpoint_threshold_must_belong_to_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sae.pt'
            model = SAE(input_dim=4, hidden_mult=2, k=1)
            torch.save({'state_dict': model.state_dict(), 'meta': {
                'threshold': .25, 'k': 1, 'args': {'decoder_norm': 'atom'}}}, path)
            loaded, threshold = load_baseline(path)
            self.assertEqual(threshold, .25)
            self.assertEqual(loaded.decoder_norm, 'atom')
            self.assertTrue(torch.equal(loaded.linear.weight, model.linear.weight))
            torch.save(model.upstream_state_dict(), path)
            with self.assertRaisesRegex(ValueError, 'do not reuse'):
                load_baseline(path)
            _, threshold = load_baseline(path, .5)
            self.assertEqual(threshold, .5)

    def test_export_bundle_to_runner_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.bundle(root)
            passages = np.zeros((3, 768), dtype=np.float32)
            passages[0, 0], passages[1, 1], passages[2, 2] = 1., 1., 1.
            np.save(root / 'passages.npy', passages)
            for prefix in ('train', 'val'):
                np.save(root / f'{prefix}_queries.npy', passages[:1])
            model = SAE(input_dim=768, hidden_mult=1, k=1)
            with torch.no_grad():
                model.linear.weight.copy_(torch.eye(768))
            checkpoint = root / 'sae.pt'
            torch.save({'state_dict': model.state_dict(), 'meta': {'threshold': .25, 'k': 1}}, checkpoint)
            result = subprocess.run([sys.executable, '-m', 'p1.handoff', '--data-dir', str(root),
                                     '--split-dir', str(root), '--checkpoint', str(checkpoint),
                                     '--output-dir', str(root / 'exports'), '--batch-size', '2'],
                                    check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout)['threshold'], .25)
            self.assertEqual(json.loads((root / 'exports/validation.query.json').read_text())['validation']['ids'], ['0'])
            self.assertIn('validation Q0 0 1 2', (root / 'exports/validation.run').read_text())
            rows = (root / 'exports/passages.latent.jsonl').read_text().splitlines()
            self.assertEqual(len(rows), 3)


if __name__ == '__main__':
    unittest.main()
