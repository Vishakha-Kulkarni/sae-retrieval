"""BatchTopK SAE with the released checkpoint's parameter names (linear, linear2, bias_pre).

Mirrors upstream sae/train.py SAE_bias_pre so released checkpoints load unchanged and
our checkpoints load into upstream extract_latent_concepts.py, but is device-agnostic.
"""
import torch
from torch import nn


class SAE(nn.Module):
    def __init__(self, input_dim=768, hidden_mult=32, k=32, dead_threshold=20):
        super().__init__()
        sparse_dim = input_dim * hidden_mult
        self.linear = nn.Linear(input_dim, sparse_dim, bias=False)
        self.linear2 = nn.Linear(sparse_dim, input_dim, bias=False)
        self.linear2.weight = nn.Parameter(self.linear.weight.detach().clone().T)
        self.bias_pre = nn.Parameter(torch.zeros(input_dim))
        self.k = k
        self.dead_threshold = dead_threshold
        self.register_buffer('steps_since_active', torch.zeros(sparse_dim))
        self.make_decoder_weights_norm()

    def pre_activation(self, x):
        return self.linear(x - self.bias_pre)

    def batch_topk(self, pre):
        """Keep the k * batch largest pre-activations over the whole batch (upstream BatchTopK)."""
        flat = pre.flatten()
        values, indices = torch.topk(flat, k=min(self.k * pre.shape[0], flat.numel()))
        z = torch.zeros_like(flat).scatter(0, indices, values.clamp(min=0)).view_as(pre)
        return z, values[-1].detach()

    def threshold(self, pre, threshold):
        """Inference-time encoding used by upstream extraction: keep activations above threshold."""
        return pre * (pre > threshold)

    def decode(self, z):
        return self.linear2(z) + self.bias_pre

    def forward(self, x, track_dead=True):
        pre = self.pre_activation(x)
        z, min_act = self.batch_topk(pre)
        recon = self.decode(z)
        dead_recon = None
        if track_dead and self.training:
            with torch.no_grad():
                active = (z > 0).any(0)
                self.steps_since_active.add_(1).mul_(~active)
            dead = self.steps_since_active > self.dead_threshold
            if dead.any():
                masked = pre.masked_fill(~dead, float('-inf'))
                values, indices = torch.topk(masked, k=min(2 * self.k, pre.shape[1]), dim=1)
                values = torch.where(torch.isinf(values), torch.zeros_like(values), values)
                dead_recon = self.linear2(torch.zeros_like(pre).scatter(1, indices, values))
        return recon, z, dead_recon, min_act

    @torch.no_grad()
    def make_decoder_weights_norm(self):
        # Upstream normalises along dim=-1 (rows of the 768 x sparse matrix); kept for checkpoint parity.
        self.linear2.weight.data /= self.linear2.weight.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def project_decoder_grad(self):
        """Remove the gradient component parallel to the unit-norm decoder rows (upstream convention)."""
        w, g = self.linear2.weight, self.linear2.weight.grad
        if g is not None:
            g -= (g * w).sum(-1, keepdim=True) * w

    def upstream_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if k != 'steps_since_active'}


def load_sae(path, k=32, map_location='cpu'):
    """Load a released upstream state_dict or a P2 checkpoint dict."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    state = payload.get('state_dict', payload)
    meta = payload.get('meta', {})
    input_dim, sparse_dim = state['linear.weight'].shape[1], state['linear.weight'].shape[0]
    model = SAE(input_dim, sparse_dim // input_dim, meta.get('k', k))
    model.load_state_dict({k: v for k, v in state.items() if k != 'steps_since_active'}, strict=False)
    return model, meta
