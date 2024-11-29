# flash attention pytorch implementation
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

# 执行标准的pytorch softmax和attention计算
expected_softmax = torch.softmax(Q_mat @ K_mat.T, dim=1)
expected_attention = expected_softmax @ V_mat


# 分块（tiling）尺寸，以SRAM的大小计算得到
Br = 4
Bc = d

# flash attention算法流程的第2步，首先在HBM中创建用于存储输出结果的O，全部初始化为0
O = torch.zeros((N, d))
# flash attention算法流程的第2步，用来存储softmax的分母值，在HBM中创建
l = torch.zeros((N, 1))
# flash attention算法流程的第2步，用来存储每个block的最大值，在HBM中创建
m = torch.full((N, 1), -torch.inf)

# 算法流程的第5步，执行外循环
for block_start_Bc in range(0, N, Bc):
    block_end_Bc = block_start_Bc + Bc
    # line 6, load a block from matmul input tensor
    # 算法流程第6步，从HBM中load Kj, Vj的一个block到SRAM
    Kj = K_mat[block_start_Bc:block_end_Bc, :]  # shape Bc x d
    Vj = V_mat[block_start_Bc:block_end_Bc, :]  # shape Bc x d
    # 算法流程第7步，执行内循环
    for block_start_Br in range(0, N, Br):
        block_end_Br = block_start_Br + Br
                # 算法流程第8行，从HBM中分别load以下几项到SRAM中
        mi = m[block_start_Br:block_end_Br, :]  # shape Br x 1
        li = l[block_start_Br:block_end_Br, :]  # shape Br x 1
        Oi = O[block_start_Br:block_end_Br, :]  # shape Br x d
        Qi = Q_mat[block_start_Br:block_end_Br, :]  # shape Br x d

        # 算法流程第9行
        Sij = Qi @ Kj.T  # shape Br x Bc

        # 算法流程第10行，计算当前block每行的最大值
        mij_hat = torch.max(Sij, dim=1).values[:, None]

        # 算法流程第10行，计算softmax的分母
        pij_hat = torch.exp(Sij - mij_hat)
        lij_hat = torch.sum(pij_hat, dim=1)[:, None]

        # 算法流程第11行，找到当前block的每行最大值以及之前的最大值
        mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None]

        # 算法流程第11行，计算softmax的分母，但是带了online计算的校正，此公式与前面说的online safe softmax不一致，但是是同样的数学表达式，只是从针对标量的逐个计算扩展到了针对逐个向量的计算
        li_new = torch.exp(mi - mi_new) * li + torch.exp(mij_hat - mi_new) * lij_hat

        # 算法流程第12行，计算每个block的输出值
        Oi = (li * torch.exp(mi - mi_new) * Oi / li_new) + (torch.exp(mij_hat - mi_new) * pij_hat / li_new) @ Vj

        # 算法流程第13行
        m[block_start_Br:block_end_Br, :] = mi_new  # row max
        l[block_start_Br:block_end_Br, :] = li_new  # softmax denominator
        # 算法流程第12行，将Oi再写回到HBM
        O[block_start_Br:block_end_Br, :] = Oi  # 每次内循环需要写到HBM中

print(torch_snr_error(expected_attention, O))