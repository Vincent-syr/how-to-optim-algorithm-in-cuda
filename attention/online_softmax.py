import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from utils import torch_snr_error

def standard_softmax(x: torch.tensor):
    n_dim = x.shape[0]
    max_val = torch.tensor(float('-inf'))
    ans = torch.empty_like(x)
    sum_val = 0
    for i in range(n_dim):
        max_val = torch.max(max_val, x[i])
    for i in range(n_dim):
        sum_val += torch.exp(x[i] - max_val)
    for i in range(n_dim):
        ans[i] = torch.exp(x[i] - max_val) / sum_val
    return ans

def online_softmax(x: torch.tensor):
    n_dim = x.shape[-1]
    max_val = torch.tensor(float('-inf'))
    ans = torch.empty_like(x)
    sum_val = 0

    for i in range(n_dim):
        pre_max = max_val
        max_val = torch.max(max_val, x[i])
        sum_val = sum_val * torch.exp(pre_max - max_val) + torch.exp(x[i] - max_val)
    for i in range(n_dim):
        ans[i] = torch.exp(x[i] - max_val) / sum_val
    return ans

    
def tiling_softmax(x: torch.tensor, n_split: int):    
    def m_func(x):
        local_dim = x.shape[-1]
        local_max = torch.tensor(float('-inf'))
        for i in range(local_dim):
            local_max = torch.max(local_max, x[i])
        return local_max

    def f_func(x, local_max):
        local_dim = x.shape[-1]
        fx = torch.empty(local_dim)
        local_sum = 0
        for i in range(local_dim):
            fx[i] = torch.exp(x[i] - local_max)
            local_sum += fx[i]
        return fx, local_sum
    
    n_dim = x.shape[-1]
    ans = torch.empty_like(x)
    local_dim = n_dim // n_split
    global_max = torch.tensor(float("-inf"))

    local_max = torch.randn(n_split)
    local_sum = torch.randn(n_split)
    fx = torch.randn(n_split, local_dim)

    # 可并行算
    for i in range(n_split):
        local_x = x[local_dim * i : local_dim * (i+1)]
        local_max[i] = m_func(local_x)
        fx[i, :], local_sum[i] = f_func(local_x, local_max[i])
        global_max = torch.max(global_max, local_max[i])

    global_sum = 0
    # 汇总算global, 可并行
    for i in range(n_split):
        global_sum += local_sum[i] * torch.exp(local_max[i] - global_max)
        ans[local_dim * i : local_dim * (i+1)] = fx[i, :] * torch.exp(local_max[i] - global_max)

    ans = ans / global_sum
    return ans

n_dim = 6
x = torch.randn(n_dim)
expect_ans = standard_softmax(x)
print(expect_ans)

online_softmax_ans = online_softmax(x)
print(torch_snr_error(expect_ans, online_softmax_ans))
print(online_softmax_ans)

tiling_softmax_ans = tiling_softmax(x, 2)
print(torch_snr_error(expect_ans, tiling_softmax_ans))
print(tiling_softmax_ans)