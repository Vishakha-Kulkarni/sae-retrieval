import json
import tempfile
import unittest
from pathlib import Path

from p3.evaluate_index import _validate_records


class P3IndexInputTests(unittest.TestCase):
    def test_counts_documents_and_mean_concept_length(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "passages.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in (
                {"docid": "a", "ids": [1, 2], "weight": [0.7, 0.2]},
                {"docid": "b", "ids": [1], "weight": [0.3]},
            )))
            self.assertEqual(_validate_records(path), (2, 1.5))

    def test_rejects_duplicate_concepts_and_empty_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "passages.jsonl"
            path.write_text(json.dumps({"docid": "a", "ids": [1, 1], "weight": [.5, .3]}) + "\n")
            with self.assertRaisesRegex(ValueError, "unique"):
                _validate_records(path)
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "empty"):
                _validate_records(path)

    def test_rejects_duplicate_document_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "passages.jsonl"
            row = {"docid": "same", "ids": [1], "weight": [.5]}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate docid"):
                _validate_records(path)


if __name__ == "__main__":
    unittest.main()
