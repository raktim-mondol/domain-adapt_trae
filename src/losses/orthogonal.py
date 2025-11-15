import torch


def orthogonal_constraint(classifier, domain_discriminator):
    w_cls = None
    w_dom = None
    for m in classifier.modules():
        if isinstance(m, torch.nn.Linear):
            w_cls = m.weight
    for m in domain_discriminator.modules():
        if isinstance(m, torch.nn.Linear):
            w_dom = m.weight
    if w_cls is None or w_dom is None:
        return torch.tensor(0.0, device=next(classifier.parameters()).device)
    prod = w_cls @ w_dom.t()
    return torch.norm(prod, p='fro')

