# sae-retrieval

P1 implements learned additive concept scoring using frozen sparse query and
passage activations. CLI entrypoints: `python -m p1.prepare --help`,
`python -m p1.train --help`, and `python -m p1.evaluate --help`.

The official implementation is pinned under `upstream/Decoding-Dense-Embeddings`;
initialize it with `git submodule update --init --recursive` after cloning.
Real-data training and full retrieval evaluation require the team's frozen artifacts.

## P2: ranking-aware SAE training

Fine-tunes the frozen baseline SAE (B0) with one extra loss term that distils SimLM's ranking into the
concepts, then evaluates with the released CL-SR scoring. See `docs/P2_method_note.md` for the baseline
rows and the improvement criterion. Tests: `python -m unittest tests/test_p2.py tests/test_clsr.py`.

| Module | Purpose |
|---|---|
| `p2.encode`, `p2.candidates` | SimLM embeddings (SimLM's titled corpus) and exact dense top-100 candidates |
| `p2.sae`, `p2.train` | BatchTopK SAE (upstream objective; `--decoder-norm atom` = the paper's per-feature norm) and ranking-aware fine-tuning (`--rank-space recon\|latent\|bm25`, `--rank-loss listwise\|naacl\|none`) |
| `p2.clsr` | Released CL-SR BM25 scoring, vectorised; verified against upstream `latent_retrieve.py` |
| `p2.evaluate` | NMSE, paper Table 1 protocol (`table1`), latent export in upstream format |
| `p2.concepts` | Concept stability of a fine-tuned SAE against B0 |

Run order on NSCC (all settings in `scripts/nscc/env.sh`):

```bash
qsub scripts/nscc/reencode_titled.pbs                 # embeddings + candidates (data_titled)
qsub -v NAME=B0_final_atom,DECODER_NORM=atom scripts/nscc/baseline.pbs   # B0
PROJECT_ID=... bash scripts/nscc/submit_sweep.sh      # B1 control, NAACL, ours (validation selection)
bash scripts/nscc/evaluate_run.sh B0 $BASELINE        # dev + TREC DL: Table 1, CL-SR
bash scripts/nscc/evaluate_run.sh listwise_bm25_w0.1_s42                 # same for a P2 run (+ concepts)
```

## P3: adaptive passage sparsity

P3's implementation and method notes are in [`p3/README.md`](p3/README.md).
The encoder adapter in `p3/adaptive_encoder.py` retains thresholded SAE
activations with minimum and maximum per-passage counts and emits the sparse
rows expected by the upstream CL-SR pipeline. `p3/extract_passage_latents.py`
provides adaptive and fixed-k extraction modes; `python -m p3.validation_sweep`
screens threshold/min/max settings on validation dense candidates.
`python -m p3.evaluate_index` builds the pinned upstream CL-SR index and reports
full retrieval metrics, serialized index size, and query latency for validation
or final evaluation.

Tune the adaptive threshold on the training-derived validation split only. The
P3 experiment still needs the team's frozen SAE checkpoint, passage embeddings,
validation/evaluation setup, and index/search measurements. Do not use MS MARCO
dev or TREC DL to tune the threshold.
