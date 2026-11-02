"""Do fine-tuned concepts still mean what B0's concepts meant? python -m p2.concepts --help

Latent i of a model fine-tuned from B0 starts as B0's latent i, so concepts can be compared by index:
- top-passage overlap: Jaccard of each concept's top-N activating passages under B0 vs the new model
  (high = the concept still fires on the same text, so B0's description still applies);
- passage overlap: share of each passage's B0 concepts that stay active under the new model.
B0 compared with itself gives 1.0 for both, which bounds the scale.
"""
import argparse
import json

import numpy as np
import torch

from .encode import default_device
from .sae import load_sae


@torch.no_grad()
def codes(model, threshold, x, device, batch_size=4096):
    out = []
    for start in range(0, len(x), batch_size):
        batch = torch.as_tensor(np.array(x[start:start + batch_size]), dtype=torch.float32, device=device)
        out.append(model.threshold(model.pre_activation(batch), threshold).cpu())
    return torch.cat(out)


def compare(z_ref, z_new, top_n=50, min_fires=None):
    """z_*: [passages, concepts] activations of the same passages under two models."""
    min_fires = top_n if min_fires is None else min_fires
    active_ref, active_new = z_ref > 0, z_new > 0
    # Only concepts that fire often enough in the sample to have a top-N list under B0.
    keep = torch.nonzero(active_ref.sum(0) >= min_fires).flatten()
    k = min(top_n, z_ref.shape[0])
    top_ref = torch.topk(z_ref[:, keep], k, dim=0).indices.T
    top_new = torch.topk(z_new[:, keep], k, dim=0).indices.T
    jaccard = []
    for a, b in zip(top_ref.tolist(), top_new.tolist()):
        a, b = set(a), set(b)
        jaccard.append(len(a & b) / len(a | b))
    retained = (active_ref & active_new).sum(1).float() / active_ref.sum(1).clamp(min=1).float()
    jaccard = np.array(jaccard)
    return {'concepts_compared': int(len(keep)),
            'top_passage_jaccard_mean': float(jaccard.mean()),
            'top_passage_jaccard_median': float(np.median(jaccard)),
            'share_concepts_jaccard_ge_0.5': float((jaccard >= 0.5).mean()),
            'passage_concepts_retained_mean': float(retained.mean()),
            'active_per_passage_ref': float(active_ref.sum(1).float().mean()),
            'active_per_passage_new': float(active_new.sum(1).float().mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', required=True, help='B0 checkpoint')
    parser.add_argument('--checkpoint', required=True, help='Fine-tuned checkpoint (initialised from --reference)')
    parser.add_argument('--passages', required=True)
    parser.add_argument('--sample', type=int, default=50_000, help='Dense codes: ~5 GB per model at 50K')
    parser.add_argument('--top-n', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=default_device())
    args = parser.parse_args()
    passages = np.load(args.passages, mmap_mode='r')
    rows = np.sort(np.random.default_rng(args.seed).choice(len(passages), min(args.sample, len(passages)),
                                                           replace=False))
    x = passages[rows]
    results = {}
    for name, path in (('reference', args.reference), ('checkpoint', args.checkpoint)):
        model, meta = load_sae(path, map_location=args.device)
        model.to(args.device).eval()
        results[name] = codes(model, meta['threshold'], x, args.device)
        if results[name].shape[1] != results.get('reference', results[name]).shape[1]:
            parser.error('Checkpoints have different dictionary sizes; concepts cannot be compared by index')
    print(json.dumps({'reference': args.reference, 'checkpoint': args.checkpoint, 'passages': len(rows),
                      **compare(results['reference'], results['checkpoint'], args.top_n)}, indent=1))


if __name__ == '__main__':
    main()
