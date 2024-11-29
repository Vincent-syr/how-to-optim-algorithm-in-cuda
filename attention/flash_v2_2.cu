
/*
CUDA MODE提示注意点：
    1. shared memory size
    2. register size
    3. 如何使用PTX和Godbolt检测溢出
*/

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

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: determine Bc, Br dynamically
    constexpr int Br = 32, Bc = 32;
    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);
    const int Tc = ceil((float) N / Bc); 
    const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device); 
    m = m.to(device);
    O = O.to(device);

    // // Calculate SRAM size needed per block
    // const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    // int max_sram_size;
    // cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    // printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);
    // dim3 grid_dim(B, nh);  // batch_size x num_heads
    // dim3 block_dim(Bc);  // Bc threads per block

    dim3 grid_dim(1);
    dim3 block_dim(32, 32);
    float scaling = 1 / sqrt(d);
    FlashAttnNaive<Br, Bc, d><<<grid_dim, block_dim>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        scaling, N, Tr, Tc, O.data_ptr<float>()
    );
    return O;
}