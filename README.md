# Ranking-Aware Concept-Level Sparse Retrieval with Sparse Autoencoders

Code for the AI6103 (NTU) course project *Ranking-Aware Concept-Level Sparse Retrieval with Sparse
Autoencoders*. We reproduce CL-SR Efficient ([Park, Kim and Ko, EMNLP 2025](https://github.com/Tro-fish/Decoding-Dense-Embeddings))
on MS MARCO, then change one pipeline stage at a time and measure each change against the reproduction (B0):

| Package | Method (report section) | Stage changed | Dev MRR@10 (B0 = 0.316) |
|---|---|---|---|
| [`p1/`](p1/) | Learned impact scoring | scorer | **0.360 ± 0.002** |
| [`p2/`](p2/) | Ranking-aware SAE training | SAE | **0.328 ± 0.002** |
| [`p3/`](p3/) | Adaptive passage sparsity | passage index | negative result (validation only) |
| [`p4/`](p4/) | Learned concept pruning | concept vocabulary | 0.313 at 23.5% fewer index entries (DF: 0.299) |
| [`p5/`](p5/) | Query concept expansion | query | 0.320 (untrained threshold control: 0.322) |

Our best systems beat the published CL-SR Efficient on dev MRR@10 (0.361 vs 0.343), TREC DL19 NDCG@10 (0.650
vs 0.643) and DL20 NDCG@10 (0.626 vs 0.593). The tables and significance tests are in the report.

## Setup

```bash
git clone <this repo> && cd sae-retrieval
git submodule update --init --recursive      # pinned upstream code: upstream/Decoding-Dense-Embeddings
pip install -r requirements.txt pytrec_eval   # PyTorch 2.2+
python -m unittest discover -s tests          # CPU, seconds, no data needed
```

P4 and the BM25 mismatch subsets also need Pyserini (Python ≥ 3.12, Java 21), which is easiest in a separate
environment; see [`p4/README.md`](p4/README.md) or [`scripts/nscc/bm25_setup.sh`](scripts/nscc/bm25_setup.sh).

## Reproducing the results

[`scripts/reproduce.sh`](scripts/reproduce.sh) runs everything, one stage at a time. Each step skips its
output if it already exists, so you can rerun a stage after a timeout.

```bash
DRY_RUN=1 bash scripts/reproduce.sh all           # print every command first
WORK=/scratch/sae bash scripts/reproduce.sh data  # then run stage by stage
WORK=/scratch/sae bash scripts/reproduce.sh b0
WORK=/scratch/sae bash scripts/reproduce.sh scoring sae sparsity pruning expansion
```

| Stage | What it does | Needs | Reproduces |
|---|---|---|---|
| `test` | Unit tests | CPU | — |
| `smoke` | B0 pipeline and the query-expansion rows on a 100k-passage slice ([`scripts/smoke.sh`](scripts/smoke.sh)) | laptop | wiring only |
| `data` | MS MARCO download, the shared split (checked against [`splits/SHA256SUMS`](splits/SHA256SUMS)), TREC DL, SimLM embeddings of the titled corpus, dense top-100 candidates and teacher scores | GPU, ~12 h | — |
| `mismatch` | BM25 (k1 = 0.9, b = 0.4) runs and the vocabulary-mismatch subsets. Optional: the subsets are committed in [`splits/p5_mismatch_k0.9_b0.4/`](splits/p5_mismatch_k0.9_b0.4/) | Pyserini | Mis@1k subset (988 queries) |
| `b0` | B0 SAE (228k steps), NMSE and reconstructed dense MRR@10, concept export, released CL-SR scoring on dev, DL19 and DL20, and the shared index for query expansion | GPU, ~28 h | Table 2, B0 row of Table 3 |
| `scoring` | Learned scorer (seeds 42–44), validation-tuned BM25 grid (28 settings), held-out evaluation | GPU | Table 4, Table 3 scoring rows |
| `sae` | Validation sweep (B1, Kang et al., recon, latent, CL-SR space × λ), three seeds of the selected run, concept stability | GPU, ~8 h per run | Table 5, Table 3 SAE rows |
| `sparsity` | Top-128 activation cache, fixed-k indexes (32/48/64/128), adaptive index, full-index validation | GPU | adaptive-sparsity paragraph |
| `pruning` | Lucene impact index, gate training (λ ∈ {0.1, 0.4, 0.8, 1.6}, seeds 0–2), DF baseline, pruned indexes, search | GPU + Pyserini | Figure 2, Table 3 pruning rows |
| `expansion` | Expander training (3 seeds), validation rows B0/T/N/L, Holm family, one-shot dev/DL runs | GPU | Table 3 expansion rows |

You can set all paths and choices with environment variables, documented at the top of the script: `WORK`,
`RAW`, `DATA`, `PY`, `BM25_PY`, `DEVICE`. Validation choices default to the report's selections (`SAE_SEL=listwise_bm25_w0.2`,
`T_ROW=T0.5`, `N_ROW=N4`). To follow the protocol strictly, run a stage's validation part, read its outputs,
and set those variables before the final runs. Dev and TREC DL are never used for selection.

On the NSCC cluster, the same steps exist as separate PBS jobs under [`scripts/nscc/`](scripts/nscc/) (settings
in [`env.sh`](scripts/nscc/env.sh)). The learned-scoring runs on the TC2 cluster are under [`scripts/tc2/`](scripts/tc2/).

**Not scripted:** the "Combined" rows of Table 3 (single seed-42 runs that stack methods on the ranking-aware
SAE) and the k2 = 15 recalibration check in the B0 and P2 paragraphs. To run the k2 check, build a
`p1.index` from the SAE's passage latents and run `python -m p1.retrieve --k2 15 …` with no `--checkpoint`.

## Repository layout

```
p1/        learned impact scoring: data prep, 193-parameter MLP scorer, GPU posting-list retrieval, locked held-out eval
p2/        SimLM encoding and candidates, BatchTopK SAE (fixed port of upstream), ranking-aware fine-tuning,
           vectorised released CL-SR scorer (p2.clsr), Table 1 protocol, concept stability
p3/        adaptive / fixed-k passage encoders, sharded on-disk indexes and evaluators
p4/        Pyserini (Lucene) CL-SR index, gate training data, gate, DF baseline, search
p5/        candidate pool, expander, upstream-exact BM25, row evaluator, mismatch subsets, Holm-corrected tests
scripts/   reproduce.sh (start here), smoke.sh, split.sh, bm25.sh, p5.sh, cluster jobs (nscc/, tc2/)
splits/    the frozen validation split (IDs + checksums) and the BM25 mismatch subsets
tests/     unit tests, including score-for-score parity with upstream latent_retrieve.py
docs/      per-method notes (P2, P3, P5), shared-artifact contract, report sources
experiments/p1/, p4/results/, sae-simlm-k32-baseline/…/p3_full_eval/   committed result files (small JSON)
upstream/  official CL-SR code, pinned as a git submodule
```

Method notes with the full selection protocol and the recorded numbers:
[`docs/P2_method_note.md`](docs/P2_method_note.md), [`p3/README.md`](p3/README.md) and
[`docs/P3_results.md`](docs/P3_results.md), [`p4/README.md`](p4/README.md),
[`docs/P5_method_note.md`](docs/P5_method_note.md), and [`docs/shared_artifacts.md`](docs/shared_artifacts.md)
for the files every method shares.

## Data and protocol

- **Corpus:** MS MARCO passages (8,841,823), encoded from SimLM's *titled* corpus, as the released pipeline does.
- **Split:** the labelled train queries are split into 497,939 training and 5,000 validation queries (seed 42),
  shared by all methods. `scripts/split.sh` rebuilds it byte for byte and checks it against `splits/SHA256SUMS`.
- **Selection:** validation only. MS MARCO dev (6,980 queries) and TREC DL 2019/2020 (43 / 54 judged queries)
  are each run once, after selection.
- **Statistics:** trained methods report the mean ± sd over three seeds. The first seed is tested against its
  baseline with a paired randomisation test (10,000 sign flips) and Holm correction.

## Differences from the released code

These change results, and B0 uses the corrected versions:

1. **Decoder normalisation.** The released trainer normalises decoder *rows*, while the paper normalises each
   latent's decoder direction. The paper's version (`p2.train --decoder-norm atom`) cuts dead latents from 40%
   to almost none and raises reconstructed dense MRR@10 from 0.294 to 0.344.
2. **Titled corpus.** The released pipeline encodes SimLM's titled corpus, not the title-less collection
   (SimLM dense MRR@10 0.407 vs 0.393).
3. **The released SAE trainer does not run as shipped:** it has undefined variables, a `wandb` call and a
   hard-coded CUDA device. `p2/sae.py` fixes only those bugs.
4. **The released retrieval loop returns empty runs:** `indexing.py` keys postings by string and
   `latent_retrieve.py main()` looks them up by integer. It also takes ~16 s per query, so we use rank-identical
   faster scorers (`p2.clsr`, `p1.retrieve`, `p3` disk indexes, `p4` Lucene), checked against upstream in `tests/`.

## Authors

Maniar Kush Hemal, Kulkarni Vishakha Vinaykumar, Jain Riddhima, Mohit Sharma, Maheshwari Vishakha. Nanyang Technological University.
