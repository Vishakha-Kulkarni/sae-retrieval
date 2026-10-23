# P4 - CL-SR Index (Pyserini)

The paper reports index size and retrieval metrics measured with Pyserini (Table 4), but the
released repository only provides a pure-Python implementation (`cl-sr/indexing.py` and
`cl-sr/latent_retrieve.py`) that stores the inverted index as a pickle. In our tests that
implementation took about 19 seconds per query on TREC DL 2019 and produced an index of about
3 GB. The paper's latency numbers (Appendix H) were measured with a NumPy/PyTorch implementation
that is not released.

This folder rebuilds CL-SR on Pyserini (Lucene impact index). It scores passages with the same
BM25-style formula as the released `latent_retrieve.py`, so on the same concepts it gives the same
results (TREC DL 2019 with the repo SAE: NDCG@10 0.470 for both), while searching in about 25 ms
per query instead of 19 s.


## Environment Setup

```
# macOS
brew install openjdk@21
export JAVA_HOME=$(brew --prefix openjdk@21)/libexec/openjdk.jdk/Contents/Home

# Linux / Colab
sudo apt-get install -y openjdk-21-jdk-headless
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

pip install pyserini pytrec-eval-terrier orjson numpy

# check
python -c "import pyserini, pytrec_eval; print('ok')"
```

Run all commands from the folder that contains `p4/`.


## Inputs

```
# Passage concepts: one JSON line per passage, line i = passage id i
#   {"docid": 0, "ids": [8033, 20057, ...], "weight": [80.69, 16.14, ...]}
# Query concepts: one JSON dict keyed by query id
#   {"1048585": {"ids": [...], "weight": [...]}, ...}
# Both must come from the same SAE checkpoint. With P2's code:

python -m p2.evaluate export --checkpoint B0_final_200k.pt \
    --embeddings data/passages.npy --output data/passages.latent.jsonl

python -m p2.evaluate export --checkpoint B0_final_200k.pt \
    --embeddings data/dev_queries.npy --output data/dev_b0.query.json --queries

# On a Mac add --device mps and prefix the command with PYTORCH_ENABLE_MPS_FALLBACK=1

# MS MARCO dev results in the paper use the 6,980-query dev subset ("dev small").
# Keep only those queries (qrels: qrels.dev.small.tsv from MS MARCO, or
# https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt)

python -c "
import json
keep = {l.split()[0] for l in open('data/qrels.dev.small.txt')}
q = json.load(open('data/dev_b0.query.json'))
json.dump({k: v for k, v in q.items() if k in keep}, open('data/dev_small_b0.query.json', 'w'))"

# TREC DL 2019 qrels are bundled with Pyserini:
python -c "from pyserini.search import get_qrels_file; print(get_qrels_file('dl19-passage'))"
```


## CL-SR Indexing

```
# --latents      passage concepts (.jsonl), see Inputs
# --output       folder to create: index/ (Lucene), df.npy (passages per concept),
#                meta.json (postings, concepts used, index size in GB, build time)
# --max-latents  concepts kept per passage, strongest first
#                (paper: 24 = CL-SR Efficient, 65 = CL-SR Max)
# --threads      threads for Lucene indexing (default 8)
# --drop         optional .npy of concept ids to remove from every passage (pruning)
#
# BM25 passage-side settings are read from environment variables:
#   K1 (default 0.6) and B (default 1.75), the paper's values for k = 32

K1=6 B=0.75 python -m p4.pyserini_indexing \
    --latents data/passages.latent.jsonl \
    --output p4_pyserini \
    --max-latents 24
```

Takes about 10-15 minutes on a recent laptop for 8.8M passages (about 45 minutes on a Colab CPU).


## Retrieval

```
# --index    folder created by the indexing step
# --queries  query concepts (.json), see Inputs
# --output   ranking in TREC format: qid Q0 docid rank score tag (top 1000 per query)
# --qrels    relevance judgements (qid 0 docid relevance)
#            if given, prints NDCG@10, MRR@10 and Recall@1000
#            if left out, only searches and measures latency
# --repeats  full runs over all queries; latency is the fastest run
#            (default 3, as in the paper's Appendix H; use 1 for a quick check)
# --idf      repo (default, released code) or paper (Eq. 6)
#
# The query-side BM25 setting is read from an environment variable:
#   K2 (default 2.5)

K2=300 python -m p4.pyserini_search \
    --index p4_pyserini \
    --queries data/dev_small_b0.query.json \
    --output runs/dev_b0.trec \
    --qrels data/qrels.dev.small.txt
```

Latency is measured as in the paper's Appendix H: one query at a time, after 20 warm-up queries,
timing only the search call, best of 3 runs. Report the machine with the number.


## BM25 Settings

The paper's settings (k1 = 0.6, b = 1.75, k2 = 2.5) were tuned for the authors' SAE and do not
transfer to an SAE with a different activation scale. With b > 1 the length term becomes negative
for short passages. They were tuned on the validation split:

```
# val qrels from P2's files (one relevant passage per query)
python -m p4.tune_bm25 make-qrels --pos data/val.pos.npy --ids data/val_queries.ids \
    --output data/val.qrels.txt

# tries k1, then b, then k2; resumable; keeps only the best index
python -m p4.tune_bm25 run --latents data/passages.latent.jsonl \
    --queries data/val_b0.query.json --qrels data/val.qrels.txt --workdir tune_b0 \
    --k1s 0.6,3,6 --bs 0.4,0.75,1.0,1.75 --k2s 2.5,10,30,60,120,300
```


## Reference Results (B0 SAE, k = 32, top 24 concepts per passage)

```
Setting                         MRR@10   Recall@1000   Latency   Index
Paper, CL-SR Efficient          0.343    0.954         118 ms*   0.57 GB
B0, paper's BM25 settings       0.257    0.892         22.6 ms   0.674 GB
B0, tuned (k1=6, b=0.75, k2=300) 0.307   0.924         25.1 ms   0.706 GB

MS MARCO dev, 6,980 queries. *Paper: NumPy on an Intel i7-14700KF, not Lucene.
```