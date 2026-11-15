import torch
from torch import nn


class ImpactScorer(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 1), nn.Softplus())

    def forward(self, rows):
        return self.network(rows).sum()


def score_group(model, group, stats):
    from .data import features
    rows, owners = [], []
    for index, doc in enumerate(group['candidates']):
        matches = features(group['query'], doc['concepts'], stats)
        rows.extend(matches)
        owners.extend([index] * len(matches))
    device = next(model.parameters()).device
    tensor = torch.tensor(rows, dtype=torch.float32, device=device).reshape(-1, 4)
    impacts = model.network(tensor).reshape(-1)
    indices = torch.tensor(owners, dtype=torch.long, device=device)
    return impacts.new_zeros(len(group['candidates'])).index_add(0, indices, impacts)
