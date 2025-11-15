import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coeff * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, coeff=1.0):
        super().__init__()
        self.coeff = coeff

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.coeff)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, hidden_ratio=0.5, dropout=0.3):
        super().__init__()
        hidden_dim = max(1, int(in_dim * hidden_ratio))
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(in_dim, hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_dim, 1))
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)

