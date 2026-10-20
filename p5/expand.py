"""Rule-based query expansion rows, written as upstream query latent JSON: python -m p5.expand --help

python -m p5.expand neighbours --queries val.query.json --checkpoint sae.pt --stats stats.json \
    --budget 8 --output N8.val.query.json
python -m p5.expand threshold --checkpoint sae.pt --embeddings val_queries.npy --scale 0.7 \
    --output T0.7.val.query.json

neighbours (row N): add up to --budget concepts, the strongest of the --per-concept decoder neighbours of each
active query concept; an added concept's weight is its source concept's weight times their decoder cosine
(the largest over sources). Original concepts and weights are kept unchanged. --stats must be the
stats.json of the evaluated index (p1.prepare corpus --max-concepts = index --max_latents).
threshold (row T, control): re-encode queries with the SAE threshold times --scale; passages unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from p1.data import load_stats
from p2.evaluate import export
from p2.sae import load_sae
from .pool import decoder_directions, eligible, neighbour_table


def expand_neighbours(queries, model, stats, budget, per_concept=16):
    if budget == 0:
        return queries
    num_concepts = model.linear2.weight.shape[1]
    allowed = eligible(stats, num_concepts)
    indices, cosines = neighbour_table(decoder_directions(model), per_concept)
    out = {}
    for qid, query in queries.items():
        if any(not 0 <= c < num_concepts for c in query['ids']):
            raise ValueError(f'{qid}: concept outside the SAE dictionary')
        scores = {}
        for concept, weight in zip(query['ids'], query['weight']):
            for neighbour, cosine in zip(indices[concept].tolist(), cosines[concept].tolist()):
                score = weight * cosine
                if allowed[neighbour] and score > 0 and score > scores.get(neighbour, 0):
                    scores[neighbour] = score
        for concept in query['ids']:
            scores.pop(concept, None)
        added = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:budget]
        out[qid] = {'ids': list(query['ids']) + [c for c, _ in added],
                    'weight': list(query['weight']) + [w for _, w in added]}
    return out


def write_expanded(queries_path, stats_path, checkpoint, output, budget, per_concept=16, k=32):
    if Path(output).exists():
        raise FileExistsError(f'{output} exists; choose a new path')
    model, _ = load_sae(checkpoint, k)
    queries = json.loads(Path(queries_path).read_text())
    expanded = expand_neighbours(queries, model, load_stats(stats_path), budget, per_concept)
    with open(output, 'x') as target:
        json.dump(expanded, target)
    return expanded


def expand_threshold(checkpoint, embeddings_path, scale, output, threshold=None, k=32, device='cpu'):
    if Path(output).exists():
        raise FileExistsError(f'{output} exists; choose a new path')
    model, meta = load_sae(checkpoint, k)
    model.to(device).eval()
    threshold = threshold if threshold is not None else meta.get('threshold')
    if threshold is None:
        raise ValueError('Released checkpoints need --threshold (the mean_of_min_act in their filename)')
    ids_path = Path(embeddings_path).with_suffix('.ids')
    ids = ids_path.read_text().split() if ids_path.exists() else None
    export(model, threshold * scale, np.load(embeddings_path, mmap_mode='r'), output, True, ids, device)


def main():
    from p2.encode import default_device
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('neighbours', 'threshold'):
        command = commands.add_parser(name)
        command.add_argument('--checkpoint', required=True)
        command.add_argument('--k', type=int, default=32, help='Only for released upstream checkpoints')
        command.add_argument('--output', required=True, help='Query latent JSON (upstream format)')
    neighbours = commands.choices['neighbours']
    neighbours.add_argument('--queries', required=True, help='Query latent JSON from p2.evaluate export --queries')
    neighbours.add_argument('--stats', required=True, help='stats.json of the evaluated index')
    neighbours.add_argument('--budget', type=int, required=True, help='Most concepts added per query')
    neighbours.add_argument('--per-concept', type=int, default=16, help='Decoder neighbours per active concept')
    threshold = commands.choices['threshold']
    threshold.add_argument('--embeddings', required=True, help='Query .npy from p2.encode queries')
    threshold.add_argument('--scale', type=float, required=True, help='Multiplies the SAE threshold, e.g. 0.7')
    threshold.add_argument('--threshold', type=float, help='Defaults to the checkpoint mean_of_min_act')
    threshold.add_argument('--device', default=default_device())
    args = parser.parse_args()
    if args.command == 'neighbours':
        write_expanded(args.queries, args.stats, args.checkpoint, args.output, args.budget, args.per_concept, args.k)
    else:
        expand_threshold(args.checkpoint, args.embeddings, args.scale, args.output, args.threshold, args.k,
                         args.device)


if __name__ == '__main__':
    main()
