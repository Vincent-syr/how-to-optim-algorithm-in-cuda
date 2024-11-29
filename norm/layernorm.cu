#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "../utils.cuh"

// y = (x-Ex)/sqrt(Var(x)+eps) * gamma + beta
// eps = 1e-5

template<int VPT, int TPB>
__global__ void layernorm(const float* x, float* y, int normalize_shape) {
    // inlocal, out_local
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x * VPT;
    constexpr float eps = 1e-5;

    float in_local[VPT], out_local[VPT];
    copy<sizeof(float) * VPT>(x + idx, in_local);

    // reduce
    float2 loc = {0.f, 0.f}; // accumulator
    float r_normalize_shape = 1.0f / (normalize_shape);
    float value = 0.0f;
    #pragma unroll
    for (int it = 0; it < VPT; ++it) {
        value = in_local[it];
        loc.x += value * r_normalize_shape;
        loc.y += value * value * r_normalize_shape;
    }

    const float2 reduced = BlockAllReduce<SumOp, float2, TPB>(loc);

    __shared__ float mu;     // mean
    __shared__ float rsigma; // std

    if (threadIdx.x == 0) {
        mu = reduced.x;
        rsigma = rsqrt(reduced.y - mu * mu + eps);
    }
    __syncthreads();
    for (int i=0; i < VPT; ++i) {
        out_local[i] = (in_local[i] - mu) * rsigma;
    }
    copy<sizeof(float) * VPT>(out_local, y+idx);
}



int main(int argc, char const *argv[]) {
    
    constexpr int bs = 8, dim = 4096;
    thrust::host_vector<float> x_host(bs * dim);
    for (int i=0; i<x_host.size(); ++i) {
        x_host[i] = (float)(rand() % 100);
    }
    thrust::device_vector<float> x = x_host;
    thrust::device_vector<float> y(bs * dim, 0);

    constexpr int VPT = 16 / sizeof(float);
    constexpr int BPG = bs, TPB = dim / VPT;
    layernorm<VPT, TPB><<<BPG, TPB>>>(thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(y.data()), dim);

    thrust::host_vector<float> y_host = y;
    

    return 0;
}
