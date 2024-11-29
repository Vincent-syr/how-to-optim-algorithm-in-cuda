
import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils import torch_snr_error

tp_size = 4
# num_pages, page_size = 1, 2
num_pages, page_size = 3, 128
page_list = [i for i in range(num_pages)]

layers, kv_heads, tokens, head_dim, group_size = 28, 8, page_size, 32, 8
# layers, kv_heads, tokens, head_dim, group_size = 2, 2, page_size, 2, 2

page_cache_bytes = layers * 2 * kv_heads * page_size * head_dim
page_scale_bytes = page_cache_bytes // group_size * 2

def py_transpose(kv_cache, kv_scale, page_id, page_seq, tp_rank, tp_size, output):
    layers = kv_cache.size(1)
    kv_heads = kv_cache.size(3)
    tokens = kv_cache.size(4)
    head_dim = kv_cache.size(5)
    group_size = head_dim // kv_scale.size(5)
    page_size = tokens
    page_cache_bytes = layers * 2 * kv_heads * page_size * head_dim # 1835008
    page_scale_bytes = (page_cache_bytes // group_size) * 2
    
    # output = torch.empty((num_pages, (page_cache_bytes + page_scale_bytes) * tp_size), dtype=torch.int8).cuda()
    output_page = output[page_seq]
    output_cache = output_page[:page_cache_bytes * tp_size]
    output_scale = output_page[page_cache_bytes * tp_size:]

    output_cache = output_cache[page_cache_bytes * tp_rank : page_cache_bytes * (tp_rank + 1)]
    output_scale = output_scale[page_scale_bytes * tp_rank : page_scale_bytes * (tp_rank + 1)]

    page_kv_cache_gpu = kv_cache[page_id]
    page_kv_scale_gpu = kv_scale[page_id]
    # output_cache[:] = page_kv_cache_gpu.permute(2, 0, 1, 3, 4).contiguous().view(-1)
    # permuted_cache = page_kv_cache_gpu.permute(2, 0, 1, 3, 4).contiguous().view(-1)
    # # print("permuted_cache: %#x" % permuted_cache.data_ptr())
    # permuted_scale = page_kv_scale_gpu.permute(2, 0, 1, 3, 4, 5).contiguous().view(-1)
    # print("page_kv_cache_gpu: ", page_kv_cache_gpu.view(-1).tolist())
    # print("permuted_cache: ", permuted_cache.view(-1).tolist())

    # output_cache[:] = permuted_cache
    output_cache.copy_(page_kv_cache_gpu.permute(2, 0, 1, 3, 4).contiguous().view(-1))
    output_scale.copy_(page_kv_scale_gpu.permute(2, 0, 1, 3, 4, 5).contiguous().view(-1))

    # print(kv_cache[page_id].data_ptr())
    print("page_kv_cache_gpu: %#x" % page_kv_cache_gpu.data_ptr())
    print("page_kv_scale_gpu: %#x" % page_kv_scale_gpu.data_ptr())
    print("output: %#x" % output.data_ptr())
    print("output_cache: %#x" % output_cache.data_ptr())
    print("output_scale: %#x" % output_scale.data_ptr())
    print("=========================")
    return output


def py_invert_transpose(input_mem, page_id, page_seq, tp_rank, tp_size, kv_cache, kv_scale):
    layers = kv_cache.size(1)
    kv_heads = kv_cache.size(3)
    tokens = kv_cache.size(4)
    head_dim = kv_cache.size(5)
    group_size = head_dim // kv_scale.size(5)
    page_size = tokens
    page_cache_bytes = layers * 2 * kv_heads * page_size * head_dim # 1835008
    page_scale_bytes = (page_cache_bytes // group_size) * 2
    
    input_page = input_mem[page_seq]
    input_cache = input_page[:page_cache_bytes * tp_size]
    input_scale = input_page[page_cache_bytes * tp_size:]

    input_cache = input_cache[page_cache_bytes * tp_rank : page_cache_bytes * (tp_rank + 1)]
    input_scale = input_scale[page_scale_bytes * tp_rank : page_scale_bytes * (tp_rank + 1)]

    page_kv_cache_gpu = kv_cache[page_id]
    page_kv_scale_gpu = kv_scale[page_id]

    page_kv_cache_gpu.copy_(input_cache.reshape(kv_heads, layers, 2, tokens, head_dim).permute(1,2,0,3,4))
    page_kv_scale_gpu.copy_(input_scale.reshape(kv_heads, layers, 2, tokens, head_dim // group_size, 2).permute(1,2,0,3,4,5))

    
def test_transpose():   
    kv_cache_list, kv_scale_list = [], []
    for i in range(tp_size):
        kv_cache = torch.randint(-127, 127, size=[num_pages, layers, 2, kv_heads, tokens, head_dim], dtype=torch.int8).cuda()
        kv_scale = torch.randint(-127, 127, size=[num_pages, layers, 2, kv_heads, tokens, head_dim // group_size, 2], dtype=torch.int8).cuda()
        kv_cache_list.append(kv_cache)
        kv_scale_list.append(kv_scale)

    print("page_cache_bytes: ", page_cache_bytes)
    print("page_scale_bytes: ", page_scale_bytes)
    output = torch.full((num_pages, (page_cache_bytes + page_scale_bytes) * tp_size), 0, dtype=torch.int8).cuda()
    for page_seq, page_id in enumerate(page_list):
        for tp_rank in range(tp_size):
            kv_cache = kv_cache_list[tp_rank]
            kv_scale = kv_scale_list[tp_rank]
            # print("kv_cache.shape: ", kv_cache.shape)
            # print("kv_scale.shape: ", kv_scale.shape)
            # print("kv_cache: \n", kv_cache.view(-1).tolist())
            # print("kv_scale: \n", kv_scale.view(-1).tolist())
            output_tmp = py_transpose(kv_cache, kv_scale, page_id, page_seq, tp_rank, tp_size, output)
            # print("output: \n", output.tolist())
    # print("output_python: ", output_tmp.tolist())
    output_ground_truth = output_tmp.clone()
    # print(output.data_ptr())
    # print(output_tmp.data_ptr())
    # print(output_ground_truth.data_ptr())

    output = torch.full((num_pages, (page_cache_bytes + page_scale_bytes) * tp_size), 0, dtype=torch.int8).cuda()
    transpose_v0 = load(name='kv_cache_transpose', sources=['transpose_v0.cu'], extra_cuda_cflags=['-O2'])
    for page_seq, page_id in enumerate(page_list):
        for tp_rank in range(tp_size):
            kv_cache = kv_cache_list[tp_rank]
            kv_scale = kv_scale_list[tp_rank]
            output_tmp = transpose_v0.ReorderLayout3(kv_cache, kv_scale, page_id, page_seq, tp_rank, tp_size, output)
            # print("output: \n", output)
            torch.cuda.synchronize()
    # print("output_cuda: ", output_tmp.tolist())
    output_predict = output_tmp.clone()
    # print(output.data_ptr())
    # print(output_tmp.data_ptr())
    # print(output_predict.data_ptr())
    print(torch_snr_error(output_predict, output_ground_truth))


def test_invert_transpose():
    input_mem = torch.randint(-127, 127, size=(num_pages, (page_cache_bytes + page_scale_bytes) * tp_size), dtype=torch.int8).cuda()

    kv_cache_truth_list, kv_scale_truth_list = [], []
    for i in range(tp_size):
        kv_cache = torch.full((num_pages, layers, 2, kv_heads, tokens, head_dim), 0, dtype=torch.int8).cuda()
        kv_scale = torch.full((num_pages, layers, 2, kv_heads, tokens, head_dim // group_size, 2), 0, dtype=torch.int8).cuda()
        kv_cache_truth_list.append(kv_cache)
        kv_scale_truth_list.append(kv_scale)

    for page_seq, page_id in enumerate(page_list):
        for tp_rank in range(tp_size):
            kv_cache = kv_cache_truth_list[tp_rank]
            kv_scale = kv_scale_truth_list[tp_rank]
            # print("kv_cache.shape: ", kv_cache.shape)
            # print("kv_scale.shape: ", kv_scale.shape)
            # print("kv_cache: \n", kv_cache.view(-1).tolist())
            # print("kv_scale: \n", kv_scale.view(-1).tolist())
            py_invert_transpose(input_mem, page_id, page_seq, tp_rank, tp_size, kv_cache, kv_scale)
            # print("output: \n", output.tolist())
    # print("output_python: ", output_tmp.tolist())

    kv_cache_pred_list, kv_scale_pred_list = [], []
    for i in range(tp_size):
        kv_cache = torch.full((num_pages, layers, 2, kv_heads, tokens, head_dim), 0, dtype=torch.int8).cuda()
        kv_scale = torch.full((num_pages, layers, 2, kv_heads, tokens, head_dim // group_size, 2), 0, dtype=torch.int8).cuda()
        kv_cache_pred_list.append(kv_cache)
        kv_scale_pred_list.append(kv_scale)
    transpose_v0 = load(name='kv_cache_transpose', sources=['transpose_v0.cu'], extra_cuda_cflags=['-O2'])
    for page_seq, page_id in enumerate(page_list):
        for tp_rank in range(tp_size):
            kv_cache = kv_cache_pred_list[tp_rank]
            kv_scale = kv_scale_pred_list[tp_rank]
            transpose_v0.InvertReorderLayout3(input_mem, page_id, page_seq, tp_rank, tp_size, kv_cache, kv_scale)
            # print("output: \n", output)
            torch.cuda.synchronize()
    # print("output_cuda: ", output_tmp.tolist())
    # print(output.data_ptr())
    # print(output_tmp.data_ptr())
    # print(output_predict.data_ptr())
    for i in range(tp_size):
        print(torch_snr_error(kv_cache_pred_list[i], kv_cache_truth_list[i]))
        print(torch_snr_error(kv_scale_pred_list[i], kv_scale_truth_list[i]))

test_transpose()
test_invert_transpose()
