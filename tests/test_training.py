"""Small integration test; installs no data and makes no quality claims."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
class TrainingTest(unittest.TestCase):
    def test_training_checkpoint_and_evaluation(self):
        import torch
        from p1.model import ImpactScorer
        model = ImpactScorer()
        empty = model(torch.empty((0, 4)))
        self.assertEqual(empty.item(), 0.)
        empty.backward()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ('train', 'validation'):
                group = {'qid': split, 'query': {'1': 1}, 'candidates': [
                    {'docid': 'a', 'concepts': {'1': 2}, 'teacher_score': 2.},
                    {'docid': 'b', 'concepts': {'2': 1}, 'teacher_score': 0.}]}
                (root / f'{split}.jsonl').write_text(json.dumps(group) + '\n')
            (root / 'stats.json').write_text(json.dumps({
                'num_docs': 3, 'avg_doc_length': 1., 'length_definition': 'concept_count',
                'document_frequency': {'1': 1, '2': 1}}))
            (root / 'qrels').write_text('validation 0 a 1\n')
            def run(*args):
                return subprocess.run([sys.executable, '-m', *args], check=True,
                                      capture_output=True, text=True).stdout
            run('p1.train', '--train', str(root / 'train.jsonl'), '--validation',
                str(root / 'validation.jsonl'), '--stats', str(root / 'stats.json'),
                '--qrels', str(root / 'qrels'), '--output', str(root / 'model.pt'), '--epochs', '2')
            common = ['--data', str(root / 'validation.jsonl'), '--stats', str(root / 'stats.json'),
                      '--qrels', str(root / 'qrels')]
            learned = json.loads(run('p1.evaluate', *common, '--checkpoint', str(root / 'model.pt')))
            self.assertEqual(learned['mrr_at_10'], 1.)
            baseline = json.loads(run('p1.evaluate', *common, '--tune'))
            self.assertEqual(baseline['mrr_at_10'], 1.)


if __name__ == '__main__':
    unittest.main()
