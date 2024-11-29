import torch

def softmax(x: torch.tensor):
    bs, dim = x.shape[0], x.shape[1]
    y = torch.empty_like(x)
    max_val = torch.empty(bs)
    sum_val = torch.empty(bs)
    for i in range(bs):
        for j in range(dim):
            max_val[i] = max(max_val[i], x[i, j])
        for j in range(dim):
            y[i, j] = torch.exp(x[i, j] - max_val[i])
            sum_val[i] += y[i, j]
        for j in range(dim):
            y[i,j] = y[i,j] / sum_val[i]


