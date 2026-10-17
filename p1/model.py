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
    scores = []
    for doc in group['candidates']:
        rows = features(group['query'], doc['concepts'], stats)
        tensor = torch.tensor(rows, dtype=torch.float32).reshape(-1, 4)
        scores.append(model(tensor))
    return torch.stack(scores)
