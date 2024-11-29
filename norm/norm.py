import torch

"""
1. Layernorm
y = (x-Ex)/sqrt(Var(x)+eps) * gamma + beta
eps = 1e-5



2. Rmsnorm

y = y / sqrt(Ex^2 + eps) * gamma + beta

"""


def Layernorm(x: torch.tensor, axis: int = -1, eps: float = 1e-5):
    # mean
    bs, dim = x.shape[0], dim = x.shape[1]

    mean = torch.empty(bs)
    mean2 = torch.empty(bs)
    var = torch.empty(bs)
    y = torch.empty_like(x)
    for i in range(bs):
        for j in range(dim):
            mean[i] += x[i, j]
            mean2[i] += x[i, j] * x[i, j]
        mean[i] /= dim
        mean2[i] /= dim
        var[i] = mean2[i] - mean[i] * mean[i]

        for j in range(dim):
            y[i, j] = (x[i,j] - mean[i]) / torch.sqrt(var[i] + eps)
    return
    
    
