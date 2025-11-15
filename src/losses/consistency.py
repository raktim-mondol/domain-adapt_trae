import torch


def consistency_mse(p1, p2):
    return torch.mean((p1 - p2) ** 2)


def consistency_kl(p1, p2):
    eps = 1e-8
    p1 = p1.clamp(min=eps, max=1.0)
    p2 = p2.clamp(min=eps, max=1.0)
    return torch.mean(torch.sum(p1 * (torch.log(p1) - torch.log(p2)), dim=-1))

