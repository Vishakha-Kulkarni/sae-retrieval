"""Audit fixed P1 query IDs and exact/normalized text overlap without test-label tuning."""
import argparse
import json
from pathlib import Path
import re
import unicodedata


def normalize(text):
    return re.sub(r'\W+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()


def audit(train_ids, validation_ids, train_text, validation_text, dev_ids=None):
    train = Path(train_ids).read_text().split()
    validation = Path(validation_ids).read_text().split()
    train_set, val_set = set(train), set(validation)
    if len(train_set) != len(train) or len(val_set) != len(validation):
        raise ValueError('Duplicate query IDs')
    if train_set & val_set:
        raise ValueError('Train/validation query ID leakage')
    texts = {}
    for name, path, selected in [('train', train_text, train_set), ('validation', validation_text, val_set)]:
        found = {}
        with open(path) as source:
            for line in source:
                qid, text = line.rstrip('\n').split('\t', 1)
                if qid in selected:
                    found[qid] = text
        if set(found) != selected:
            raise ValueError(f'{name}: missing query text')
        texts[name] = found
    exact = set(texts['train'].values()) & set(texts['validation'].values())
    normalized = set(map(normalize, texts['train'].values())) & set(map(normalize, texts['validation'].values()))
    normalized.discard('')
    normalized_train = set(map(normalize, texts['train'].values()))
    overlap_ids = sorted(q for q, text in texts['validation'].items()
                         if normalize(text) and normalize(text) in normalized_train)
    result = {'training_queries': len(train), 'validation_queries': len(validation),
              'overlapping_query_ids': 0, 'overlapping_exact_query_texts': len(exact),
              'overlapping_normalized_query_texts': len(normalized),
              'validation_ids_with_normalized_training_text': overlap_ids,
              'scope': 'ID and text overlap audit; does not prove absence of semantic duplicates or overfitting'}
    if dev_ids:
        dev = set(Path(dev_ids).read_text().split())
        result['train_dev_overlapping_ids'] = len(train_set & dev)
        result['validation_dev_overlapping_ids'] = len(val_set & dev)
        if (train_set | val_set) & dev:
            raise ValueError('Development query IDs overlap training/validation')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('train-ids', 'validation-ids', 'train-text', 'validation-text', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--dev-ids')
    a = p.parse_args()
    result = audit(a.train_ids, a.validation_ids, a.train_text, a.validation_text, a.dev_ids)
    Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
