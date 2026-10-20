# Shared validation artifacts (owner: P5)

One job builds the files every piece tunes on: `qsub -P $PROJECT_ID scripts/nscc/shared.pbs`. It runs the
steps in dependency order, skips any output that already exists (so it can be resubmitted after a timeout),
and builds validation before train, so P1 and P3 can start before the slow train-query candidates finish.
Paths are `$RAW` and `$DATA` from `scripts/nscc/env.sh`.

| File | Built by | Used by |
|---|---|---|
| `$RAW/split/validation.{qids,queries.tsv,qrels}` | `p1.prepare split --size 5000 --seed 42` | everyone: tuning and checkpoint selection |
| `$RAW/split/train.{qids,queries.tsv,qrels}` | same command (all other labelled train queries) | P1, P2, P5 training |
| `$DATA/passages.npy` | `p2.encode passages` (row i = passage ID i) | P2, P3, candidates |
| `$DATA/val_queries.npy` + `.ids` | `p2.encode queries` | P1, P3, P5 |
| `$DATA/val.{cands,teacher,pos}.npy`, `val.run` | `p2.candidates --top-k 100` (SimLM dot products) | P1, P3, P5 |
| `$DATA/validation_*` | symlinks to the `val_*` files above | P3 (`load_split(..., "validation")`) |
| `$DATA/dev_queries.npy` + `.ids` | `p2.encode queries` on `queries.dev.small.tsv` | final runs only |
| `$DATA/train_queries.npy` + `.ids`, `train.{cands,teacher,pos}.npy`, `train.run` | as for validation | P1, P2, P5 distillation |

Formats: embeddings are float32 `[N, 768]`; `.ids` has one query ID per line in row order; `.cands.npy` is
`[Q, 100]` passage rows (= passage IDs); `.teacher.npy` holds the matching SimLM scores; `.pos.npy` is the first
judged-relevant row or -1; `.run` is a 6-column TREC run that `p1.prepare groups --teacher-run` reads. Qrels
are `qid 0 docid relevance`.

The split is deterministic: rerunning `p1.prepare split` with the same seed on the same MS MARCO train files
gives byte-identical files. The validation queries are train queries, so dev and TREC DL stay untouched until
the final runs. Never tune on `qrels.dev.small.tsv` or the TREC DL qrels.

**Before a run, check you have the frozen files:** `(cd $RAW && sha256sum -c $DATA/MANIFEST.split.sha256)`
and `(cd $DATA && sha256sum -c MANIFEST.sha256)`. Both manifests are written at the end of the job.

P3 example: `python -m p3.validation_sweep --data-dir $DATA --qrels $RAW/split/validation.qrels …`
P1 example: `python -m p1.prepare groups --teacher-run $DATA/val.run --queries <val query latents> …`
