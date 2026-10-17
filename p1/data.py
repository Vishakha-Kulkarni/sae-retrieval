"""Validated JSONL interchange format; no dependency on a particular retriever."""
import json
import math


def load_groups(path, require_teacher=False):
    groups = []
    seen = set()
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            group = json.loads(line)
            qid = str(group['qid'])
            if qid in seen:
                raise ValueError(f'{path}:{line_number}: duplicate qid {qid}')
            seen.add(qid)
            group['qid'] = qid
            group['query'] = sparse(group['query'])
            if not group['candidates']:
                raise ValueError(f'{qid}: empty candidate list')
            docids = set()
            for doc in group['candidates']:
                doc['docid'] = str(doc['docid'])
                if doc['docid'] in docids:
                    raise ValueError(f'{qid}: duplicate docid {doc["docid"]}')
                docids.add(doc['docid'])
                doc['concepts'] = sparse(doc['concepts'])
                if require_teacher or 'teacher_score' in doc:
                    if not math.isfinite(float(doc['teacher_score'])):
                        raise ValueError(f'{qid}: nonfinite teacher score')
            groups.append(group)
    if not groups:
        raise ValueError(f'{path}: empty dataset')
    return groups


def sparse(values):
    result = {}
    for key, value in values.items():
        concept = int(key)
        activation = float(value)
        if concept < 0 or activation < 0 or not math.isfinite(activation):
            raise ValueError('Concept IDs must be nonnegative; activations finite and nonnegative')
        if concept in result:
            raise ValueError('Duplicate normalized concept ID')
        if activation:
            result[concept] = activation
    return result


def load_stats(path):
    with open(path) as source:
        stats = json.load(source)
    n = stats['num_docs']
    avg = stats['avg_doc_length']
    if not isinstance(n, int) or n <= 0 or not math.isfinite(avg) or avg <= 0:
        raise ValueError('Corpus size and mean sum of activations must be positive')
    stats['document_frequency'] = {int(k): v for k, v in stats['document_frequency'].items()}
    if any(k < 0 or not isinstance(v, int) or not 0 <= v <= n
           for k, v in stats['document_frequency'].items()):
        raise ValueError('Document frequencies must be integers within [0, num_docs]')
    return stats


def idf(concept, stats):
    df = stats['document_frequency'].get(concept)
    if df is None:
        raise ValueError(f'Corpus statistics missing concept {concept}')
    return math.log1p((stats['num_docs'] - df + 0.5) / (df + 0.5))


def features(query, document, stats):
    """Each row belongs to one shared concept; empty intersections score zero."""
    length = sum(document.values()) / stats['avg_doc_length']
    return [[math.log1p(query[c]), math.log1p(document[c]), idf(c, stats),
             math.log1p(length)] for c in sorted(query.keys() & document.keys())]


def bm25(query, document, stats, k1=1.2, b=0.75):
    length = sum(document.values()) / stats['avg_doc_length']
    return sum(query[c] * idf(c, stats) * document[c] * (k1 + 1) /
               (document[c] + k1 * (1 - b + b * length))
               for c in query.keys() & document.keys())
