#include <stdint.h>
#include <torch/types.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
// #include <device_functions.h>
#include <vector_types.h>

#include <float.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>

// kv_cache_reorder

struct Stride {
    int p_stride_cache;
    int l_stride_cache;
    int kv_stride_cache;
    int h_stride_cache;
    int t_stride_cache;
    int d_stride_cache;

    int p_stride_scale;
    int l_stride_scale;
    int kv_stride_scale;
    int h_stride_scale;
    int t_stride_scale;
    int d_stride_scale;
};

__global__ void reorder(
    char* kv_cache_gpu,
    char* kv_scale_gpu,
    const Stride src_stride,
    const Stride dst_stride,
    int64_t* page_ids_device,
    int page_nums,
    int num_elems,
    int head_dim,
    int group_size,
    int tp_rank,
    int tp_size,
    int outer_start_page,
    char* outer_mem
) {
    int idx_l = blockIdx.y;
    int idx_kv = threadIdx.y;
    int idx_h = blockIdx.z / page_nums;
    int idx_t = blockIdx.x;
    int idx_d = threadIdx.x;
    
    int page_id_offset = blockIdx.z % page_nums;
    int page_id = page_ids_device[page_id_offset];

    int page_cache_bytes = src_stride.p_stride_cache;
    int page_scale_bytes = src_stride.p_stride_scale;
    char* page_kv_cache_gpu = kv_cache_gpu + page_id * page_cache_bytes;
    char* page_kv_scale_gpu = kv_scale_gpu + page_id * page_scale_bytes;

    int src_cache_idx = idx_l * src_stride.l_stride_cache + idx_kv * src_stride.kv_stride_cache + idx_h * src_stride.h_stride_cache + idx_t * src_stride.t_stride_cache + idx_d * src_stride.d_stride_cache;

    if (src_cache_idx >= num_elems) {
        return;
    }

    int dst_cache_idx = idx_d * dst_stride.d_stride_cache + idx_t * dst_stride.t_stride_cache + idx_kv * dst_stride.kv_stride_cache + idx_l * dst_stride.l_stride_cache + idx_h * dst_stride.h_stride_cache;

    char* outer_cache = outer_mem + (page_cache_bytes + page_scale_bytes) * tp_size * (outer_start_page + page_id_offset) + tp_rank * page_cache_bytes;
    outer_cache[dst_cache_idx] = page_kv_cache_gpu[src_cache_idx];

    if (idx_d >= head_dim / group_size * 2) {
        return;
    }
    int src_scale_idx = idx_l * src_stride.l_stride_scale + idx_kv * src_stride.kv_stride_scale + idx_h * src_stride.h_stride_scale + idx_t * src_stride.t_stride_scale + idx_d * src_stride.d_stride_scale;

    int dst_scale_idx = idx_d * dst_stride.d_stride_scale + idx_t * dst_stride.t_stride_scale + idx_kv * dst_stride.kv_stride_scale + idx_l * dst_stride.l_stride_scale + idx_h * dst_stride.h_stride_scale;

    char* outer_scale = outer_cache + (tp_size - tp_rank) * page_cache_bytes + tp_rank * page_scale_bytes;

    outer_scale[dst_scale_idx] = page_kv_scale_gpu[src_scale_idx];
}

torch::Tensor ReorderLayout3(
    const torch::Tensor& kv_cache, 
    const torch::Tensor& kv_scale,
    const torch::Tensor& page_ids,
    int tp_rank,
    int tp_size,
    int outer_start_page,
    torch::Tensor& outer_mem
) {
    const int num_elems =  kv_cache.numel() / kv_cache.size(0);
    const int layers = kv_cache.size(1);
    const int kv_heads = kv_cache.size(3);
    const int tokens = kv_cache.size(4);
    const int head_dim = kv_cache.size(5);
    const int group_size = head_dim / kv_scale.size(5);
    const int page_size = tokens;
    const int page_nums = page_ids.size(0);

    const dim3 block(head_dim, 2);
    const dim3 grid(tokens, layers, kv_heads * page_nums);

    Stride src_stride;
    src_stride.p_stride_cache = layers * 2 * kv_heads * tokens * head_dim;
    src_stride.l_stride_cache = 2 * kv_heads * tokens * head_dim;
    src_stride.kv_stride_cache = kv_heads * tokens * head_dim;
    src_stride.h_stride_cache = tokens * head_dim;
    src_stride.t_stride_cache = head_dim;
    src_stride.d_stride_cache = 1;

    src_stride.p_stride_scale = layers * 2 * kv_heads * tokens * head_dim / group_size * 2;
    src_stride.l_stride_scale = 2 * kv_heads * tokens * head_dim / group_size * 2;
    src_stride.kv_stride_scale = kv_heads * tokens * head_dim / group_size * 2;
    src_stride.h_stride_scale = tokens * head_dim / group_size * 2;
    src_stride.t_stride_scale = head_dim / group_size * 2;
    src_stride.d_stride_scale = 1;

    std::cout << "p_stride_cache: " << src_stride.p_stride_cache << std::endl;
    std::cout << "l_stride_cache: " << src_stride.l_stride_cache << std::endl;
    std::cout << "kv_stride_cache: " << src_stride.kv_stride_cache << std::endl;
    std::cout << "h_stride_cache: " << src_stride.h_stride_cache << std::endl;
    std::cout << "t_stride_cache: " << src_stride.t_stride_cache << std::endl;
    std::cout << "d_stride_cache: " << src_stride.d_stride_cache << std::endl;

    Stride dst_stride;
    dst_stride.p_stride_cache = kv_heads * layers * 2 * tokens * head_dim;
    dst_stride.h_stride_cache = layers * 2 * tokens * head_dim;
    dst_stride.l_stride_cache = 2 * tokens * head_dim;
    dst_stride.kv_stride_cache = tokens * head_dim;
    dst_stride.t_stride_cache = head_dim;
    dst_stride.d_stride_cache = 1;

    dst_stride.p_stride_scale = kv_heads * layers * 2 * tokens * head_dim / group_size * 2;
    dst_stride.h_stride_scale = layers * 2 * tokens * head_dim / group_size * 2;
    dst_stride.l_stride_scale = 2 * tokens * head_dim / group_size * 2;
    dst_stride.kv_stride_scale = tokens * head_dim / group_size * 2;
    dst_stride.t_stride_scale = head_dim / group_size * 2;
    dst_stride.d_stride_scale = 1;

    // const int page_cache_bytes = layers * 2 * kv_heads * page_size * head_dim * sizeof(int8_t);
    // const int page_scale_bytes = layers * 2 * kv_heads * page_size * head_dim / group_size * 2;

    std::cout << std::endl;
    std::cout << "outer_start_page: " << outer_start_page << std::endl;
    std::cout << "layers: " << layers << std::endl;
    std::cout << "kv_heads: " << kv_heads << std::endl;
    std::cout << "head_dim: " << head_dim << std::endl;
    std::cout << "group_size: " << group_size << std::endl;
    // std::cout << "page_cache_bytes: " << page_cache_bytes << std::endl;
    // std::cout << "page_scale_bytes: " << page_scale_bytes << std::endl;
    std::cout << "tp_rank: " << tp_rank << std::endl;
    std::cout << "tp_size: " << tp_size << std::endl;
    std::cout << "num_elems: " << num_elems << std::endl;

    std::cout << "===========================" << std::endl;
    reorder<<<grid, block>>>(
        (char*)(kv_cache.data_ptr<int8_t>()),
        (char*)(kv_scale.data_ptr<int8_t>()),
        src_stride,
        dst_stride,
        page_ids.data_ptr<int64_t>(),
        page_nums,
        num_elems,
        head_dim,
        group_size,
        tp_rank,
        tp_size,
        outer_start_page,
        (char*)outer_mem.data_ptr<int8_t>()
    );
    return outer_mem;
}


__global__ void invert_reorder(
    char* outer_mem,
    const Stride src_stride,
    const Stride dst_stride,
    int outer_start_page,
    int64_t* page_ids_device,
    int page_nums,
    int num_elems,
    int head_dim,
    int group_size,
    int tp_rank,
    int tp_size,
    char* kv_cache_gpu,
    char* kv_scale_gpu
) {
    int idx_l = blockIdx.y;
    int idx_kv = threadIdx.y;
    int idx_h = blockIdx.z / page_nums;
    int idx_t = blockIdx.x;
    int idx_d = threadIdx.x;

    int page_id_offset = blockIdx.z % page_nums;
    int page_id = page_ids_device[page_id_offset];

    int page_cache_bytes = src_stride.p_stride_cache;
    int page_scale_bytes = src_stride.p_stride_scale;
    char* page_kv_cache_gpu = kv_cache_gpu + page_id * page_cache_bytes;
    char* page_kv_scale_gpu = kv_scale_gpu + page_id * page_scale_bytes;

    int src_cache_idx = idx_l * src_stride.l_stride_cache + idx_kv * src_stride.kv_stride_cache + idx_h * src_stride.h_stride_cache + idx_t * src_stride.t_stride_cache + idx_d * src_stride.d_stride_cache;

    if (src_cache_idx >= num_elems) {
        return;
    }

    int dst_cache_idx = idx_d * dst_stride.d_stride_cache + idx_t * dst_stride.t_stride_cache + idx_kv * dst_stride.kv_stride_cache + idx_l * dst_stride.l_stride_cache + idx_h * dst_stride.h_stride_cache;

    char* input_cache = outer_mem + (page_cache_bytes + page_scale_bytes) * tp_size * (outer_start_page + page_id_offset) + tp_rank * page_cache_bytes;
    page_kv_cache_gpu[dst_cache_idx] = input_cache[src_cache_idx];

    if (idx_d >= head_dim / group_size * 2) {
        return;
    }

    int src_scale_idx = idx_l * src_stride.l_stride_scale + idx_kv * src_stride.kv_stride_scale + idx_h * src_stride.h_stride_scale + idx_t * src_stride.t_stride_scale + idx_d * src_stride.d_stride_scale;

    int dst_scale_idx = idx_d * dst_stride.d_stride_scale + idx_t * dst_stride.t_stride_scale + idx_kv * dst_stride.kv_stride_scale + idx_l * dst_stride.l_stride_scale + idx_h * dst_stride.h_stride_scale;

    char* input_scale = input_cache + (tp_size - tp_rank) * page_cache_bytes + tp_rank * page_scale_bytes;

    page_kv_scale_gpu[dst_scale_idx] = input_scale[src_scale_idx];
}


void InvertReorderLayout3(
    const torch::Tensor& outer_mem, 
    const torch::Tensor& page_ids,
    int outer_start_page,
    int tp_rank,
    int tp_size,
    torch::Tensor& kv_cache,
    torch::Tensor& kv_scale
) {
    const int num_elems =  kv_cache.numel() / kv_cache.size(0);

    const int layers = kv_cache.size(1);
    const int kv_heads = kv_cache.size(3);
    const int tokens = kv_cache.size(4);
    const int head_dim = kv_cache.size(5);
    const int group_size = head_dim / kv_scale.size(5);
    const int page_size = tokens;
    const int page_nums = page_ids.size(0);

    const dim3 block(head_dim, 2);
    const dim3 grid(tokens, layers, kv_heads * page_nums);

    Stride src_stride;
    src_stride.p_stride_cache = kv_heads * layers * 2 * tokens * head_dim;
    src_stride.h_stride_cache = layers * 2 * tokens * head_dim;
    src_stride.l_stride_cache = 2 * tokens * head_dim;
    src_stride.kv_stride_cache = tokens * head_dim;
    src_stride.t_stride_cache = head_dim;
    src_stride.d_stride_cache = 1;

    src_stride.p_stride_scale = kv_heads * layers * 2 * tokens * head_dim / group_size * 2;
    src_stride.h_stride_scale = layers * 2 * tokens * head_dim / group_size * 2;
    src_stride.l_stride_scale = 2 * tokens * head_dim / group_size * 2;
    src_stride.kv_stride_scale = tokens * head_dim / group_size * 2;
    src_stride.t_stride_scale = head_dim / group_size * 2;
    src_stride.d_stride_scale = 1;
    std::cout << "l_stride_cache: " << src_stride.l_stride_cache << std::endl;
    std::cout << "kv_stride_cache: " << src_stride.kv_stride_cache << std::endl;
    std::cout << "h_stride_cache: " << src_stride.h_stride_cache << std::endl;
    std::cout << "t_stride_cache: " << src_stride.t_stride_cache << std::endl;
    std::cout << "d_stride_cache: " << src_stride.d_stride_cache << std::endl;

    Stride dst_stride;
    dst_stride.p_stride_cache = layers * 2 * kv_heads * tokens * head_dim;
    dst_stride.l_stride_cache = 2 * kv_heads * tokens * head_dim;
    dst_stride.kv_stride_cache = kv_heads * tokens * head_dim;
    dst_stride.h_stride_cache = tokens * head_dim;
    dst_stride.t_stride_cache = head_dim;
    dst_stride.d_stride_cache = 1;

    dst_stride.p_stride_scale = layers * 2 * kv_heads * tokens * head_dim / group_size * 2;
    dst_stride.l_stride_scale = 2 * kv_heads * tokens * head_dim / group_size * 2;
    dst_stride.kv_stride_scale = kv_heads * tokens * head_dim / group_size * 2;
    dst_stride.h_stride_scale = tokens * head_dim / group_size * 2;
    dst_stride.t_stride_scale = head_dim / group_size * 2;
    dst_stride.d_stride_scale = 1;
    // const int page_cache_bytes = layers * 2 * kv_heads * page_size * head_dim * sizeof(int8_t);
    // const int page_scale_bytes = layers * 2 * kv_heads * page_size * head_dim / group_size * 2;

    std::cout << "outer_start_page: " << outer_start_page << std::endl;
    std::cout << "num_elems: " << num_elems << std::endl;
    std::cout << "layers: " << layers << std::endl;
    std::cout << "kv_heads: " << kv_heads << std::endl;
    std::cout << "head_dim: " << head_dim << std::endl;
    std::cout << "group_size: " << group_size << std::endl;
    std::cout << "tp_rank: " << tp_rank << std::endl;
    std::cout << "tp_size: " << tp_size << std::endl;
    std::cout << "num_elems: " << num_elems << std::endl;

    std::cout << "===========================" << std::endl;
    invert_reorder<<<grid, block>>>(
        (char*)outer_mem.data_ptr<int8_t>(),
        src_stride,
        dst_stride,
        outer_start_page,
        page_ids.data_ptr<int64_t>(),
        page_nums,
        num_elems,
        head_dim,
        group_size,
        tp_rank,
        tp_size,
        (char*)(kv_cache.data_ptr<int8_t>()),
        (char*)(kv_scale.data_ptr<int8_t>())
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ReorderLayout3", torch::wrap_pybind_function(ReorderLayout3), "ReorderLayout3");

    m.def("InvertReorderLayout3", torch::wrap_pybind_function(InvertReorderLayout3), "InvertReorderLayout3");
}