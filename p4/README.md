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

## Full Index

 Builds the unpruned index. Also writes df.npy, the document frequency of each concept: the number of passages it appears in (counted over each passage's top 24 concepts). Common, stopword-like concepts have a high df. The full index prevents the gate from droping any concepts.

```
python -m p4.pyserini_indexing --latents passages.latent.jsonl --output idx/full
```


## Gate Training Data

For every training question, this takes the 100 passages SimLM ranked the highest and works out the individual CL-SR points each shared concept adds to that passage's score. The result is one sparse matrix with a row per question-passage pair and a column per concept. Adding up a row gives the normal CL-SR score of that passage. We use the same BM-25 settings as the index.

```
python -m p4.prepare_data --latents passages.latent.jsonl --queries train.query.json \
    --ids train_queries.ids --cands train.cands.npy --out gate_X.npz
```


## Gate Training

We train the model using a results from a teacher simlm model. We calculate the CL-SR score of each of the 100 candidate passages, with every concept multiplied with a gate which is responsible for deciding whether the concept is active or not. This gate is trained such that it follows the same scoring distribution as the teacher model. Concepts whose values are below 0.5 are dropped. -lam decides the amount of concepts we are dropping.

```
python -m p4.gate_train --x gate_X.npz --teacher train.teacher.npy --df idx/full/df.npy \
    --lam 0.4 --out gate_lam0.4
```


## DF Baseline

This part creates the baseline df to compare the gate with. The code checks the amount dropped by the gate and removes the most common concepts until the index is about the same size as the one produced by the learnt gate. This helps us establish a baseline of whether dropping common concepts performs better than a learnt gate.

```
python -m p4.df_drops --df idx/full/df.npy --gates "gate_lam*.drop.npy"
```


## Pruned Index

This is the same script we use for the full index, but with --drop pointing to a gate or DF drop list. We keep the length of each passage the same as before dropping, so the concepts that are left keep their original scores. Only the dropped concepts are removed.


```
python -m p4.pyserini_indexing --latents passages.latent.jsonl --output idx/gate_lam0.4 \
    --drop gate_lam0.4.drop.npy
```


## Search

This runs the questions through an index and reports MRR@10, NDCG@10, Recall@1000, index size and latency
```
python -m p4.pyserini_search --index idx/gate_lam0.4 --queries dev.query.json \
    --qrels qrels.dev.small.tsv --output runs/dev_gate_lam0.4.trec
```