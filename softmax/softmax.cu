#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "../utils.cuh"

template<int WARP_WIDTH, int VPT>
__global__ void warp_softmax(const float* __restrict__ x, int dim, float* __restrict__ y) {
    /*
    1个warp处理一行，适用于cols<1024的场景, 每个th最多32个data
    */  
    int warp_id = threadIdx.y, lane_id = threadIdx.x;
    int cols_per_thread = (dim + WARP_WIDTH - 1) / WARP_WIDTH;
    int row_id = blockIdx.x * blockDim.y;
    const float* row_x = x + row_id * dim;
    float* row_y = y + row_id * dim;
    // 1. max
    float thread_max = row_x[0];
    int i = 0, col_id = lane_id * cols_per_thread + i;
    for (col_id = lane_id * cols_per_thread + i; i < cols_per_thread && col_id < dim; ++i) {
        thread_max = max(thread_max, row_x[col_id]);
    }
    int row_max = WarpAllReduce<MaxOp, float, WARP_WIDTH>(thread_max);
    
    // 2. sum
    int thread_sum = 0;
    i = 0, col_id = lane_id * cols_per_thread + i;
    for (col_id = lane_id * cols_per_thread + i; i<cols_per_thread && col_id < dim; ++i) {
        row_y[col_id] = exp(row_x[col_id] - row_max);
        thread_sum += row_y[col_id];
    }
    int row_sum = WarpAllReduce<SumOp, float, WARP_WIDTH>(thread_sum);
    
    // 3. result
    i = 0, col_id = lane_id * cols_per_thread + i;
    for (col_id = lane_id * cols_per_thread + i; i<cols_per_thread && col_id < dim; ++i) {
        row_y[col_id] /= row_sum;        
    }
     
}

template<int TPB, int VPT>
__global__ void block_softmax( const float* __restrict__  x, int dim,  float* __restrict__  y) {
    /*
    一个block处理一行，适用于1024 < cols < 4096的场景，需要借助shared memory
    shared memory, 
    */
    // 1. block reduce max
    const int idx = blockIdx.x * dim + threadIdx.x * VPT;
    float local_in[VPT], local_out[VPT];
    copy<sizeof(float)*VPT>(x + idx, local_in);
    float thread_max = local_in[0];
    #pragma unroll
    for (int i=0; i<VPT; ++i) {
        thread_max = max(local_in[i], thread_max);
    }
    const float row_max = BlockAllReduce<MaxOp, float, TPB>(thread_max);

    // 2. block reduce sum, y[i] = reduceSum(), 
    float thread_sum = 0.f;
    #pragma unroll
    for (int i=0; i<VPT; ++i) {
        local_out[i] = exp(local_in[i] - row_max);
        thread_sum += local_out[i];
    }
    const float row_sum = BlockAllReduce<SumOp, float, TPB>(thread_sum);
    
    // 3. result
    for (int i=0; i<VPT; ++i) {
        local_out[i] = local_out[i] / row_sum;
    }
    copy<sizeof(float) * VPT>(local_out, y + idx);
}


void test_warp_softamx() {
    constexpr int bs = 8, dim = 512;
    thrust::host_vector<float> h_x(bs * dim);
    for (int i=0; i<h_x.size(); ++i) {
        h_x[i] = (float)(rand() % 100);
    }
    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y(bs * dim, 0);

    constexpr int VPT = 16 / sizeof(float);
    constexpr dim3 TPB = (32, 8);    // warp_size, num_warps 
    const int BPG = bs / 8;         // 
    float* d_x_ptr = thrust::raw_pointer_cast(d_x.data());
    float* d_y_ptr = thrust::raw_pointer_cast(d_y.data());
    warp_softmax<32, VPT><<<BPG, TPB>>>(d_x_ptr, dim, d_y_ptr);
}

void test_block_softamx() {

    constexpr int bs = 8, dim = 4096;
    thrust::host_vector<float> h_x(bs * dim);
    for (int i=0; i<h_x.size(); ++i) {
        h_x[i] = (float)(rand() % 100);
    }
    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y(bs * dim, 0);

    float* d_x_ptr = thrust::raw_pointer_cast(d_x.data());
    float* d_y_ptr = thrust::raw_pointer_cast(d_y.data());

    constexpr int VPT = 16 / sizeof(float), TPB = 256;
    const int BPG = bs;
    int shared_mem_bytes = dim * sizeof(float);

    block_softmax<TPB, VPT><<<BPG, TPB>>>(d_x_ptr, dim, d_y_ptr);
}



int main(int argc, char const *argv[]) {
    
    constexpr int bs = 8, dim = 4096;
    thrust::host_vector<float> h_x(bs * dim);
    for (int i=0; i<h_x.size(); ++i) {
        h_x[i] = (float)(rand() % 100);
    }
    test_block_softamx();
    test_warp_softamx();

    return 0;
}
