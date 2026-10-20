# Shared validation artifacts (owner: P5)

Every piece tunes on the same files. They come from two places:

| You need | Get it with | Where |
|---|---|---|
| Validation split: 5,000 train queries, their text and qrels | `OUT=~/msmarco_raw bash scripts/split.sh` (any machine, ~30 MB download, under a minute, no GPU) | `$OUT/split/validation.{qids,queries.tsv,qrels}` |
| Training queries and qrels (the other 497,939) | same script | `$OUT/split/train.{qids,queries.tsv,qrels}` |
| Dev queries (6,980) and qrels, for final runs only | same script | `$OUT/queries.dev.small.tsv`, `$OUT/qrels.dev.small.tsv` |
| Query embeddings, dense top-100 candidate lists, SimLM teacher scores | `qsub -P $PROJECT_ID scripts/nscc/shared.pbs` (NSCC GPU, once for the team) | shared drive `$DATA` (below) |

Git holds only the split's definition: `splits/validation.qids` (the 5,000 IDs) and `splits/SHA256SUMS`
(checksums of the MS MARCO inputs and every split file). `split.sh` rebuilds the split and fails if any file
differs, so everyone gets byte-identical files without MS MARCO data in git. Point `$RAW` at `$OUT`, or copy
`$OUT/split` to `$RAW/split`; `shared.pbs` reuses it and stops if its split differs from `splits/validation.qids`.

## What `shared.pbs` writes to `$DATA`

Validation first, so P1 and P3 can start before the slow train-query candidates finish. Steps whose output
exists are skipped, so the job can be resubmitted after a timeout.

| File | Contents | Used by |
|---|---|---|
| `passages.npy` | SimLM passage embeddings, float32 `[8.8M, 768]`, row i = passage ID i | P2, P3 |
| `val_queries.npy`, `val_queries.ids` | validation query embeddings `[5000, 768]`; IDs in row order | P1, P3, P5 |
| `val.cands.npy`, `val.teacher.npy` | dense top-100 passage rows `[5000, 100]` and their SimLM scores | P1, P3, P5 |
| `val.pos.npy`, `val.run` | first relevant row (-1 if none); the same top-100 as a 6-column TREC run | P1 (`p1.prepare groups --teacher-run`), P5 |
| `validation_*` | links to the `val_*` files, the names P3's `load_split(..., "validation")` reads | P3 |
| `dev_queries.npy`, `dev_queries.ids` | dev query embeddings | final runs only |
| `train_queries.npy`, `train.{cands,teacher,pos}.npy`, `train.run` | as for validation, 497,939 queries | P1, P2, P5 training |
| `MANIFEST.sha256`, `MANIFEST.split.sha256` | checksums of all of the above | everyone, before a run |

Check before a run: `(cd $DATA && sha256sum -c MANIFEST.sha256)`.

## Rules and details

- Tune and select only on the validation split. Dev and TREC DL qrels are for the one-shot final runs.
- Validation shares no query with train or dev. The split qrels have 10 fewer lines than `qrels.train.tsv`
  because that file contains 10 exact duplicate lines.
- `queries.dev.small.tsv` isn't in MS MARCO's `queries.tar.gz`; both scripts derive it as the
  `queries.dev.tsv` rows whose IDs appear in `qrels.dev.small.tsv`.

P3: `python -m p3.validation_sweep --data-dir $DATA --qrels $RAW/split/validation.qrels …`
P1: `python -m p1.prepare groups --teacher-run $DATA/val.run --queries <validation query latents> …`
