"""Exercise the frozen-export handoff without external data or model downloads."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
class RunnerTest(unittest.TestCase):
    def test_export_to_validation_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            passages = [dict(docid=d, ids=[c], weight=[1.])
                        for d, c in [('a', 1), ('b', 2), ('c', 3)]]
            (root / 'passages').write_text(''.join(json.dumps(p) + '\n' for p in passages))
            command = [sys.executable, '-m', 'p1.run', '--passages', str(root / 'passages'),
                       '--output-dir', str(root / 'out'), '--top-k', '2', '--epochs', '1',
                       '--validation-qrels', str(root / 'qrels')]
            for split in ('train', 'validation'):
                (root / f'{split}.queries').write_text(json.dumps({split: dict(ids=[1], weight=[1.])}))
                (root / f'{split}.ids').write_text(split + '\n')
                (root / f'{split}.run').write_text(
                    f'{split} Q0 a 1 2 simlm\n{split} Q0 b 2 0 simlm\n')
                for option, suffix in [('queries', 'queries'), ('run', 'run'), ('qids', 'ids')]:
                    command.extend([f'--{split}-{option}', str(root / f'{split}.{suffix}')])
            (root / 'qrels').write_text('validation 0 a 1\n')
            subprocess.run(command, check=True, capture_output=True, text=True)
            for method in ('released_bm25', 'tuned_bm25', 'learned'):
                metrics = json.loads((root / 'out' / f'{method}.metrics.json').read_text())
                self.assertEqual(metrics['mrr_at_10'], 1.)
                self.assertEqual(metrics['num_queries'], 1)
                self.assertTrue((root / 'out' / f'{method}.run').is_file())
            # Re-running cannot overwrite a completed experiment.
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('FileExistsError', result.stderr)
            # Reject split leakage before creating any output artifacts.
            (root / 'validation.ids').write_text('train\n')
            command[command.index('--output-dir') + 1] = str(root / 'leaked')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('disjoint', result.stderr)
            self.assertFalse((root / 'leaked').exists())
