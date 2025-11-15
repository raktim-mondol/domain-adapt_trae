import torch


def entropy_loss_from_logits(logits):
    probs = torch.softmax(logits, dim=-1)
    return entropy_loss(probs)


def entropy_loss(probs):
    eps = 1e-8
    p = probs.clamp(min=eps, max=1.0)
    return (-p * torch.log(p)).sum(dim=-1).mean()

