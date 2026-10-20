# sae-retrieval

P1 implements learned additive concept scoring using frozen sparse query and
passage activations. CLI entrypoints: `python -m p1.prepare --help`,
`python -m p1.train --help`, and `python -m p1.evaluate --help`.

The official implementation is pinned under `upstream/Decoding-Dense-Embeddings`;
initialize it with `git submodule update --init --recursive` after cloning.
Real-data training and full retrieval evaluation require the team's frozen artifacts.

## P2: ranking-aware SAE training

Fine-tunes the released SAE with one extra loss term that distils SimLM's ranking into the concepts,
then evaluates the result with the unchanged upstream CL-SR. See `docs/P2_method_note.md` for the
baseline rows and the improvement criterion. Entrypoints: `python -m p2.encode`, `p2.candidates`,
`p2.train`, `p2.evaluate`; NSCC jobs are in `scripts/nscc/`. Tests: `python -m unittest tests/test_p2.py`.
