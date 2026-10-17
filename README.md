# sae-retrieval

P1 implements learned additive concept scoring using frozen sparse query and
passage activations. CLI entrypoints: `python -m p1.prepare --help`,
`python -m p1.train --help`, and `python -m p1.evaluate --help`.

The official implementation is pinned under `upstream/Decoding-Dense-Embeddings`;
initialize it with `git submodule update --init --recursive` after cloning.
Real-data training and full retrieval evaluation require the team's frozen artifacts.
