"""Validated JSONL interchange format; no dependency on a particular retriever."""
import json
import math


def load_groups(path, require_teacher=False):
    return list(IndexedGroups(path, require_teacher))


def validate_group(group, require_teacher=False):
    group['qid'] = str(group['qid'])
    group['query'] = sparse(group['query'])
    if not group['candidates']:
        raise ValueError(f'{group["qid"]}: empty candidate list')
    docids = set()
    for doc in group['candidates']:
        doc['docid'] = str(doc['docid'])
        if doc['docid'] in docids:
            raise ValueError(f'{group["qid"]}: duplicate docid {doc["docid"]}')
        docids.add(doc['docid'])
        doc['concepts'] = sparse(doc['concepts'])
        if require_teacher or 'teacher_score' in doc:
            if not math.isfinite(float(doc['teacher_score'])):
                raise ValueError(f'{group["qid"]}: nonfinite teacher score')
    return group


class IndexedGroups:
    """Store query byte offsets rather than all candidate activations in memory."""
    def __init__(self, path, require_teacher=False):
        self.path, self.require_teacher = path, require_teacher
        self.offsets, self.qids = [], set()
        with open(path, 'rb') as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                group = validate_group(json.loads(line), require_teacher)
                qid = group['qid']
                if qid in self.qids:
                    raise ValueError(f'{path}: duplicate qid {qid}')
                self.qids.add(qid)
                self.offsets.append(offset)
        if not self.offsets:
            raise ValueError(f'{path}: empty dataset')

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        offset = self.offsets[index]
        with open(self.path, 'rb') as source:
            source.seek(offset)
            return validate_group(json.loads(source.readline()), self.require_teacher)


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
    if stats.get('length_definition') != 'concept_count':
        raise ValueError('Regenerate statistics with p1.prepare: length_definition must be concept_count')
    if not isinstance(n, int) or n <= 0 or not math.isfinite(avg) or avg <= 0:
        raise ValueError('Corpus size and mean concept count must be positive')
    stats['document_frequency'] = {int(k): v for k, v in stats['document_frequency'].items()}
    if any(k < 0 or not isinstance(v, int) or not 0 <= v <= n
           for k, v in stats['document_frequency'].items()):
        raise ValueError('Document frequencies must be integers within [0, num_docs]')
    return stats


def idf(concept, stats):
    df = stats['document_frequency'].get(concept)
    if df is None:
        raise ValueError(f'Corpus statistics missing concept {concept}')
    return math.log((stats['num_docs'] - df + 0.5) / (df + 0.5))


def features(query, document, stats):
    """Each row belongs to one shared concept; empty intersections score zero."""
    length = len(document) / stats['avg_doc_length']
    return [[math.log1p(query[c]), math.log1p(document[c]), idf(c, stats),
             math.log1p(length)] for c in sorted(query.keys() & document.keys())]


def bm25(query, document, stats, k1=0.6, b=2.0, k2=1.5):
    length = len(document) / stats['avg_doc_length']
    return sum((query[c] * (k2 + 1) / (query[c] + k2)) * idf(c, stats) * document[c] * (k1 + 1) /
               denominator(document[c], length, k1, b)
               for c in query.keys() & document.keys())


def denominator(activation, relative_length, k1, b):
    value = activation + k1 * (1 - b + b * relative_length)
    if value == 0:
        raise ValueError('Released BM25 denominator is zero; this parameter setting is undefined')
    return value
