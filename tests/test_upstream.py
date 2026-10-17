import ast
from collections import defaultdict
import json
import gzip
import math
from pathlib import Path
import pickle
import tempfile
import unittest

from p1.data import bm25, load_groups, load_stats
from p1.prepare import build_corpus, convert_collection, latent, prepare_groups, split_queries


UPSTREAM = Path(__file__).resolve().parents[1] / 'upstream/Decoding-Dense-Embeddings/cl-sr'


def official_functions(filename, namespace):
    # Execute the pinned official functions without importing GPU/CLI dependencies.
    path = UPSTREAM / filename
    tree = ast.parse(path.read_text())
    functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    exec(compile(functions, str(path), 'exec'), namespace)
    return namespace


class UpstreamTests(unittest.TestCase):
    def test_raw_collection_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'collection.tsv').write_text('0\tfirst passage\n1\tsecond passage\n')
            convert_collection(root / 'collection.tsv', root / 'passages.jsonl.gz', 1)
            with gzip.open(root / 'passages.jsonl.gz', 'rt') as source:
                rows = [json.loads(line) for line in source]
            self.assertEqual(rows, [{'id': '0', 'title': '', 'contents': 'first passage'}])

    def test_released_index_and_scoring_parity(self):
        indexing = official_functions('indexing.py', {
            'json': json, 'math': math, 'pickle': pickle, 'defaultdict': defaultdict,
            'tqdm': lambda values, **kw: values})
        scoring = official_functions('latent_retrieve.py', {
            'k1': .6, 'k2': 1.5, 'b': 2., 'TOP_K': 1000, 'defaultdict': defaultdict})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            passages = root / 'passages.jsonl'
            records = [{'docid': 'a', 'ids': [1, 2, 3], 'weight': [.7, .2, .1]},
                       {'docid': 'b', 'ids': [1, 3], 'weight': [.8, .5]},
                       {'docid': 'c', 'ids': [1], 'weight': [.4]}]
            passages.write_text(''.join(json.dumps(r) + '\n' for r in records))
            stats = build_corpus(passages, root / 'corpus.db', root / 'stats.json', 2)
            indexing['build_inverted_index'](passages, root / 'official.pkl', 2)
            with open(root / 'official.pkl', 'rb') as source:
                index = pickle.load(source)
            self.assertEqual(stats['num_docs'], index['total_docs'])
            self.assertEqual(stats['avg_doc_length'], index['avgdl'])
            query = {1: .6, 2: .3, 3: .8}
            official = dict(scoring['retrieve_one'](
                [str(c) for c in query], list(query.values()), index['inverted_index'],
                index['document_lengths'], index['avgdl']))
            for record in records:
                self.assertAlmostEqual(bm25(query, latent(record, 2), stats), official[record['docid']])
            self.assertLess(index['inverted_index']['1']['idf'], 0)
            load_stats(root / 'stats.json')

    def test_adapter_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'passages').write_text(json.dumps({'docid': 'd', 'ids': [3], 'weight': [.9]}) + '\n')
            build_corpus(root / 'passages', root / 'corpus.db', root / 'stats')
            (root / 'queries').write_text(json.dumps({'q': {'ids': [3], 'weight': [.7]}}))
            (root / 'teacher').write_text('q Q0 d 1 2.5 simlm\n')
            prepare_groups(root / 'queries', root / 'corpus.db', root / 'teacher', root / 'groups', 1)
            group = load_groups(root / 'groups', require_teacher=True)[0]
            self.assertEqual(group['candidates'][0]['teacher_score'], 2.5)
            self.assertEqual(group['query'], {3: .7})
            (root / 'raw_queries').write_text('1\tone\n2\ttwo\n3\tthree\n4\tfour\n')
            (root / 'qrels').write_text('1 0 a 1\n2 0 b 1\n3 0 c 1\n4 0 d 1\n')
            for name in ('split1', 'split2'):
                split_queries(root / 'raw_queries', root / 'qrels', root / name, 2, 42)
            validation = (root / 'split1/validation.qids').read_text()
            self.assertEqual(validation, (root / 'split2/validation.qids').read_text())
            train = set((root / 'split1/train.qids').read_text().split())
            self.assertFalse(train & set(validation.split()))
            self.assertEqual(len(train | set(validation.split())), 4)

    def test_reject_bad_latents_and_old_statistics(self):
        for record in ({'ids': [1], 'weight': []}, {'ids': [1, 1], 'weight': [1, 2]},
                       {'ids': [1], 'weight': [0]}):
            with self.assertRaises(ValueError):
                latent(record)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stats'
            path.write_text(json.dumps({'num_docs': 1, 'avg_doc_length': 1, 'document_frequency': {}}))
            with self.assertRaisesRegex(ValueError, 'Regenerate'):
                load_stats(path)


if __name__ == '__main__':
    unittest.main()
