import numba.cuda
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils import torch_snr_error
import numba
import math

torch.manual_seed(456)

N, d = 64, 128
Q_mat = torch.rand((N, d), device="cuda")
K_mat = torch.rand((N, d), device="cuda")
V_mat = torch.rand((N, d), device="cuda")
scalar_factor = 1 / math.sqrt(d)

def standard_attn(Q, K, V):
    S = Q @ K.T * scalar_factor                                 # (N, N)
    m, _ = torch.max(S, dim=-1, keepdim=True)    # (N)
    l = torch.sum(torch.exp(S - m), dim=-1, keepdim=True)  # (N)  row sum
    P = torch.exp(S - m) / l      # (N, N)
    O = P @ V
    return O

N_inp, N_out = N, N
Br = 16
Bc = 16
Tc = (N_inp + Bc - 1) // Bc # (4)
Tr = (N_out + Br - 1) // Br # (4)    

def flash_attn_v2(Q, K, V):
    O = torch.zeros((N, d), device="cuda") # global mem
    scalar_factor = 1 / math.sqrt(d)

    # 分块（tiling）尺寸，以SRAM的大小计算得到
    # 分配shared mem
    Qi = torch.zeros((Br, d), device="cuda")
    Kj = torch.zeros((Bc, d), device="cuda")
    Vj = torch.zeros((Bc, d), device="cuda")
    Oi = torch.zeros((Br, d), device="cuda")
    li = torch.zeros((Br, 1), device="cuda")  # row sum, shape Br x 1
    mi = torch.full((Br, 1), fill_value=-torch.inf, device="cuda")  # shape Br x 1
    # 算法流程第3步，执行外循环, 已经是各自thread执行
    for block_start_Br in range(0, N, Br):
        block_end_Br = block_start_Br + Br
        # 算法流程第4步，从HBM中load Qi 的一个block到SRAM
        Qi = Q_mat[block_start_Br:block_end_Br, :]  # [Br, d]
        # 算法流程第5步，初始化每个block的值
        mi = torch.full((Br, 1), fill_value=-torch.inf, device="cuda")  # shape Br x 1

        # 算法流程第6步，执行内循环
        for block_start_Bc in range(0, N, Bc):
            block_end_Bc = block_start_Bc + Bc

            # 算法流程第7步，load Kj, Vj到SRAM
            Kj = K_mat[block_start_Bc:block_end_Bc, :]  # [Bc, d]
            Vj = V_mat[block_start_Bc:block_end_Bc, :]  # [Bc, d]

            # 算法流程第8步
            Sij = Qi @ Kj.T * scalar_factor # [Br, Bc]
            # 算法流程第9步
            mi_new = torch.maximum(mi, Sij.max(dim=-1, keepdim=True).values)
            Pij_hat = torch.exp(Sij - mi_new)
            li = torch.exp(mi - mi_new) * li + torch.sum(Pij_hat, dim=1)[:, None]
            # 算法流程第10步
            Oi = Oi * torch.exp(mi - mi_new) + Pij_hat @ Vj # [Br, d]
            mi = mi_new

        # 第12步
        Oi = Oi / li

        # 第14步
        O[block_start_Br:block_end_Br, :] = Oi
    return O

# def flash_attn_v2_2(Q, K, V):
#     """
#         类似cuda实现，for循环版本
#         Q,K,V: (N, d),
#     """
#     O = torch.zeros((N, d)) # global mem, output
#     scalar_factor = 1 / math.sqrt(d)

#     # 分块（tiling）尺寸，以SRAM的大小计算得到
#     # 分配shared mem
#     Qi = torch.zeros((Br, d))
#     Kj = torch.zeros((Bc, d))
#     Vj = torch.zeros((Bc, d))
#     # 拆分的
#     for block_start_Br in range(0, N, Br):
#         # smem
#         Qi = Q_mat[block_start_Br:block_end_Br, :]  # [Br, d]
#         # 算法流程第6步，执行内循环
#         for block_start_Bc in range(0, N, Bc):
#             block_end_Bc = block_start_Bc + Bc
#             # 算法流程第7步，load Kj, Vj到SRAM
#             Kj = K_mat[block_start_Bc:block_end_Bc, :]  # [Bc, d]
#             Vj = V_mat[block_start_Bc:block_end_Bc, :]  # [Bc, d]

#             # for dd in d:
#                 # local_q = Qi[]



    

# # def flash_attn_v2_decoding(Q, K, V):
# #     """
# #         seqlenq != seqlenkv时，包括了decoding的情况
# #     """
# #     N_inp = Q.shape[0]
# #     N_out = K.shape[0]
# #     Tc = (N_inp + Bc - 1) // Bc
# #     Tr = (N_out + Br - 1) // Br
# #     scalar_factor = 1 / math.sqrt(d)
# #     for i in range(Tr):


    

@numba.cuda.jit
def flash_attn_v2_numba(Q, K, V, O):
    """
        block: (x,y) 在seqlen和dim维度划分
    """
    inp_dtype = Q.dtype
    tid_x = numba.cuda.threadIdx.x
    tid_y = numba.cuda.threadIdx.y
    # on shared mem
    Qi = numba.cuda.shared.array((Br, d), dtype=inp_dtype)
    Kj = numba.cuda.shared.array((Bc, d), dtype=inp_dtype)
    Vj = numba.cuda.shared.array((Bc, d), dtype=inp_dtype)
    # actually on register, not smem
    Oi = numba.cuda.shared.array((Br, d), dtype=inp_dtype)
    li = numba.cuda.shared.array((Br, ), dtype=inp_dtype)
    mi = numba.cuda.shared.array((Br, ), dtype=inp_dtype)
    Si = numba.cuda.shared.array((Bc, ), dtype=inp_dtype)   # smem
    
    for i in range(Tr):
        # 1. load QKV to smem
        for ii in range(tid_y, Br, numba.cuda.blockDim.y):
            for dd in range(tid_x, d, numba.cuda.blockDim.x):
                Qi[ii, dd] = Q[ii + i * Br, dd]
                Oi[ii, dd] = 0
            li[ii] = 0
            mi[ii] = -math.inf
        numba.cuda.syncthreads()
        for j in range(Tc):
            for jj in range(tid_y, Bc, numba.cuda.blockDim.y):
                for dd in range(tid_x, d, numba.cuda.blockDim.x):
                    Kj[jj, dd] = K[jj + j * Bc, dd]
                    Vj[jj, dd] = V[jj + j * Bc, dd]
            numba.cuda.syncthreads()
            
            # 2. 计算, naive gemm, 存储至Si smem
            for ii in range(tid_x, Br, numba.cuda.blockDim.x):
                # gemv
                for jj in range(tid_y, Bc, numba.cuda.blockDim.y):
                    Sij = 0 # k维度累加和
                    for dd in range(d):
                        Sij += Qi[ii, dd] * Kj[jj, dd]
                    Sij *= scalar_factor
                    Si[jj] = Sij    

                numba.cuda.syncthreads()
                # mi_new = torch.maximum(mi, Sij.max(dim=-1, keepdim=True).values)
                # this need to use parallel reduction pattern
                # 其实max(m)可以融到gemm里面
                m = mi[ii]
                last_m = m
                for jj in range(Bc):
                    m = max(m, Si[jj])
                mi[ii] = m
                l = math.exp(last_m - m) * li[ii]
                
                for dd in range(d):
                    Oi[ii, dd] *= math.exp(last_m - m)
                for jj in range(Bc):
                    Si[jj] = math.exp(Si[jj] - m)
                    l += Si[jj]
                    for dd in range(d):
                        Oi[ii, dd] += Si[jj] * Vj[jj, dd]
                li[ii] = l
        # write back to global memory
        for ii in range(tid_y, Br, numba.cuda.blockDim.y):
            for dd in range(tid_x, d, numba.cuda.blockDim.x):
                O[ii + i * Br, dd] = Oi[ii, dd] / li[ii]
            # L[ii + i * Br] = li[ii]

def test_flash_attn_v2_torch():
    expected_attention = standard_attn(Q_mat, K_mat, V_mat)
    O_torch = flash_attn_v2(Q_mat, K_mat, V_mat)
    print(torch_snr_error(expected_attention, O_torch))
    # assert torch.allclose(O_torch, expected_attention)

def test_flash_attn_v2_numba():
    expected_attention = standard_attn(Q_mat, K_mat, V_mat)
    O = torch.zeros((N, d), device="cuda")
    O_torch = flash_attn_v2_numba[(32, 4), (1, )](Q_mat, K_mat, V_mat, O)
    print(torch_snr_error(expected_attention, O_torch))
    # assert torch.allclose(O_torch, expected_attention)


import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
def test_flash_attn_v2_cuda_v0():
    expected_attention = standard_attn(Q_mat, K_mat, V_mat)
    cuda_attn_v0 = load(name='minimal_attn', sources=['main.cpp', 'flash_v2.cu'], extra_cuda_cflags=['-O2'])
    O_torch = cuda_attn_v0.forward(Q_mat, K_mat, V_mat)
    print(torch_snr_error(expected_attention, O_torch))


# test_flash_attn_v2_torch()
# test_flash_attn_v2_numba()
test_flash_attn_v2_cuda_v0()
