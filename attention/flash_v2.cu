
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

template<int Br, int Bc, int d>
__global__ void FlashAttnNaive(
    const float* Q, 
    const float* K, 
    const float* V, 
    float scaling, 
    int n, 
    int Tr,
    int Tc,
    float* O
) {
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    __shared__ float Qi[Br][d];
    __shared__ float Kj[Bc][d];
    __shared__ float Vj[Bc][d];
    __shared__ float Si[Br][Bc];
    constexpr int o_per_thread_x = 1;
    constexpr int o_per_thread_y = 128 / 32;    // 4
    
    float li[o_per_thread_x];   // exp sum
    float mi[o_per_thread_x];   // max per row
    float Oi[o_per_thread_x][o_per_thread_y];
    // outer loop
    for (int i=0; i<Tr; ++i) {
        for (int ii=0; ii < o_per_thread_x; ++ii) {
            for (int dd=0; dd < o_per_thread_y; ++dd) {
                Oi[ii][dd] = 0;
            }
            li[ii] = 0.f;
            mi[ii] = -FLT_MAX;
        }
        //  load Q into smem
        for (int ii = tid_y; ii < Br; ii += blockDim.y) {
            for (int dd=tid_x; dd < d; dd += blockDim.x) {
                Qi[ii][dd] = Q[(ii + i * Br) * d + dd];
            }
        }
        // inner loop
        for (int j=0; j < Tc; ++j) {
            __syncthreads();
            // load KV to smem
            for (int jj=tid_y; jj < Bc; jj += blockDim.y) {
                for (int dd=tid_x; dd < d; dd += blockDim.x) {
                    Kj[jj][dd] = K[(jj + j*Bc) * d + dd];
                    Vj[jj][dd] = V[(jj + j*Bc) * d + dd];
                }
            }
            __syncthreads();

            // 1. gemm: compute Q @ K
            // Si = scale_factor * (Qi @ Kj.T)
            for (int ii=tid_x; ii < Br; ii += blockDim.x) {
                for (int jj=tid_y; jj < Bc; jj += blockDim.y) {
                    float Sij = 0.f;
                    for (int dd=0; dd < d; dd++) {
                        Sij += Qi[ii][dd] * Kj[jj][dd];
                    }
                    Sij *= scaling;
                    Si[ii][jj] = Sij;
                }
            }
            __syncthreads();
            // 2. online softmax and partial Oi
            for (int ii=0; ii < o_per_thread_x; ++ii) {
                float m = mi[ii];
                float last_m = m;
                for (int jj=0; jj < Bc; ++jj) {
                    if (m < Si[ii * blockDim.x + tid_x][jj]) {
                        m = Si[ii * blockDim.x + tid_x][jj];
                    }
                }
                mi[ii] = m;
                float l = expf(last_m - m) * li[ii];
                for (int dd=0; dd < o_per_thread_y; ++dd) {
                    Oi[ii][dd] *= expf(last_m - m);
                }

                for (int jj=0; jj < Bc; ++jj) {
                    float Sij = expf(Si[ii * blockDim.x + tid_x][jj] - m);
                    l += Sij;
                    for (int dd=0; dd < o_per_thread_y; ++dd) {
                        Oi[ii][dd] += Sij * Vj[jj][dd * blockDim.y + tid_y];
                    }
                }
                li[ii] = l;
            }
        }
        for (int ii=0; ii < o_per_thread_x; ++ii) {
            for (int dd = 0; dd < o_per_thread_y; ++dd) {
                int gid_x = ii * blockDim.x + tid_x + i * Br;
                int gid_y = dd * blockDim.y + tid_y;
                O[gid_x * d + gid_y] = Oi[ii][dd] / li[ii];
                // out[(ii * blockDim.x + tid_x + i * Br)]
            }
        }
    }
}




torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: determine Bc, Br dynamically
    // constexpr int Br = 32, Bc = 32;
    // const int B = Q.size(0); const int nh = Q.size(1);
    // const int N = Q.size(2); const int d = Q.size(3);

    constexpr int Br = 16, Bc = 16;
    constexpr int d = 128;
    const int N = Q.size(0);
    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    // auto l = torch::zeros({B, nh, N});
    // auto m = torch::full({B, nh, N}, -INFINITY);
    auto l = torch::zeros({N});
    auto m = torch::full({N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device); m = m.to(device);
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