"""Learned query expander (row L): a per-candidate MLP that weights pool concepts (KTD7)."""
import torch
from torch import nn
from torch.nn import functional as F

from .evaluate import file_hash

FORMAT_VERSION = 1


class Expander(nn.Module):
    def __init__(self, features, hidden=32):
        super().__init__()
        self.register_buffer('mean', torch.zeros(features))
        self.register_buffer('std', torch.ones(features))
        self.net = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        with torch.no_grad():
            self.net[-1].bias.fill_(-4.)  # softplus(-4) ~ 0.02: start close to the unexpanded query

    @torch.no_grad()
    def set_normalisation(self, features):
        self.mean.copy_(features.mean(0))
        self.std.copy_(features.std(0, unbiased=False).clamp(min=1e-6))

    def forward(self, features):
        """Non-negative added weight per pool candidate."""
        return F.softplus(self.net((features - self.mean) / self.std)).squeeze(-1)


def top_budget(weights, budget):
    """Inference weights: keep the budget largest, zero the rest."""
    keep = weights.topk(min(budget, len(weights))).indices
    return torch.zeros_like(weights).index_copy(0, keep, weights[keep])


def load_expander(path, sae_checkpoint, stats_path):
    """Refuses an SAE checkpoint or stats.json other than the ones the expander was trained with (KTD10)."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format_version') != FORMAT_VERSION:
        raise ValueError(f'{path}: expander format {payload.get("format_version")}, expected {FORMAT_VERSION}')
    for name, current in (('sae', sae_checkpoint), ('stats', stats_path)):
        if file_hash(current) != payload['input_sha256'][name]:
            raise ValueError(f'{current} differs from the {name} file this expander was trained with')
    model = Expander(len(payload['features']), payload['hidden'])
    model.load_state_dict(payload['state_dict'])
    return model.eval(), payload
