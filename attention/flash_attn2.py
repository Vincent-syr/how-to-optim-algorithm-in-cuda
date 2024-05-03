import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils import torch_snr_error

torch.manual_seed(456)

N, d = 16, 8
Q_mat = torch.rand((N, d))
K_mat = torch.rand((N, d))
V_mat = torch.rand((N, d))

expected_softmax = torch.softmax(Q_mat @ K_mat.T, dim=1)
expected_attention = expected_softmax @ V_mat

# 分块（tiling）尺寸，以SRAM的大小计算得到
Br = 4
Bc = d

O = torch.zeros((N, d))

# 算法流程第3步，执行外循环
for block_start_Br in range(0, N, Br):
    block_end_Br = block_start_Br + Br
    # 算法流程第4步，从HBM中load Qi 的一个block到SRAM
    Qi = Q_mat[block_start_Br:block_end_Br, :]
    # 算法流程第5步，初始化每个block的值
    Oi = torch.zeros((Br, d))  # shape Br x d
    li = torch.zeros((Br, 1))  # shape Br x 1
    mi = torch.full((Br, 1), -torch.inf)  # shape Br x 1

    # 算法流程第6步，执行内循环
    for block_start_Bc in range(0, N, Bc):
        block_end_Bc = block_start_Bc + Bc

        # 算法流程第7步，load Kj, Vj到SRAM
        Kj = K_mat[block_start_Bc:block_end_Bc, :]
        Vj = V_mat[block_start_Bc:block_end_Bc, :]

        # 算法流程第8步
        Sij = Qi @ Kj.T
        # 算法流程第9步
        mi_new = torch.max(torch.column_stack([mi, torch.max(Sij, dim=1).values[:, None]]), dim=1).values[:, None]
        Pij_hat = torch.exp(Sij - mi_new)
        li = torch.exp(mi - mi_new) * li + torch.sum(Pij_hat, dim=1)[:, None]
        # 算法流程第10步
        Oi = Oi * torch.exp(mi - mi_new) + Pij_hat @ Vj
        
        mi = mi_new

    # 第12步
    Oi = Oi / li

    # 第14步
    O[block_start_Br:block_end_Br, :] = Oi

print(torch_snr_error(expected_attention, O))
assert torch.allclose(O, expected_attention)
