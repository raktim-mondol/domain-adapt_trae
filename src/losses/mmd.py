import torch


def _gaussian_kernel(x, y, sigma):
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True)
    cross = x @ y.t()
    dist = x_norm + y_norm.t() - 2 * cross
    k = torch.exp(-dist / (2 * (sigma ** 2)))
    return k


def mmd_loss(z_s, z_t, bandwidths, mask_s=None, mask_t=None):
    if mask_s is not None:
        z_s = z_s[mask_s]
    if mask_t is not None:
        z_t = z_t[mask_t]
    if z_s.numel() == 0 or z_t.numel() == 0:
        return z_s.new_tensor(0.0)
    loss = z_s.new_tensor(0.0)
    for sigma in bandwidths:
        k_ss = _gaussian_kernel(z_s, z_s, sigma)
        k_tt = _gaussian_kernel(z_t, z_t, sigma)
        k_st = _gaussian_kernel(z_s, z_t, sigma)
        loss = loss + (k_ss.mean() + k_tt.mean() - 2 * k_st.mean())
    return loss

