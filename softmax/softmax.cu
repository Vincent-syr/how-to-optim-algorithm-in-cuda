#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "../utils.cuh"

// ref: https://www.oneflow.org/a/share/jishuboke/54.html

template<int WARP_SIZE, int VPT, int COLS_PER_THREAD>
__global__ void warp_softmax(const float* __restrict__ x, int dim, float* __restrict__ y) {
    /*
    1个warp处理一行，适用于cols<1024的场景, 每个th最多32个data
    */  
    int warp_id = threadIdx.y, lane_id = threadIdx.x;
    int row_id = blockIdx.x * blockDim.y;
    const float* row_x = x + row_id * dim;
    float* row_y = y + row_id * dim;

    int num_packs = COLS_PER_THREAD / VPT;
    float buf[COLS_PER_THREAD];

    // 1. max
    #pragma unroll
    for (int pack_id=0; pack_id<num_packs; ++pack_id) {
        // int col_id = pack_id * 
        int cols = (pack_id * WARP_SIZE + lane_id) * VPT;
        copy<sizeof(float) * VPT>(buf + pack_id * VPT, row_x + cols);
        float thread_max = *(buf + pack_id * VPT);
        #pragma unroll
        for (int i=0; i<VPT; ++i) {
            thread_max = max(thread_max, local_in[i]);
        }
    }
    float row_max = WarpAllReduce<MaxOp, float, WARP_SIZE>(thread_max);

    // 2. sum
    float thread_sum = 0;
    #pragma unroll
    for (int i=0; i<COLS_PER_THREAD; ++i) {
        buf[i] = exp(buf[i] - row_max);
        thread_sum += buf[i];
    }
    float warp_sum = WarpAllReduce<SumOp, float, WARP_SIZE>(thread_sum);

    // 3. compute result
    #pragma unroll
    for (int i=0; i<COLS_PER_THREAD; ++i) {
        buf[i] = buf[i] / warp_sum;
    }

    // 4. write back to gmem. TODO: 3/4可以合并到一起
    for (int pack_id=0; pack_id<num_packs; ++pack_id) {
        const int cols = (pack_id * WARP_SIZE + lane_id) * VPT;
        copy<sizeof(float) * VPT>(row_y + cols, buf + pack_id * VPT);
    }
}



template<typename T, typename ComputeType, int TPB, int VPT>
__global__ void block_smem_softmax( const T* __restrict__  x, int dim,  T* __restrict__  y) {
    /*
    一个block处理一行，适用于1024 < cols < 4096的场景，需要借助shared memory
    注意：对shared memory的赋值过程中，需要避免bank conflict 
    给 Shared memory 赋值过程中，若采用下面方法，当 pack size=2，每个线程写连续两个4 byte 地址，就会产生 Bank Conflicts。
        #pragma unroll
        for (int j = 0; j < pack_size; ++j) {
            buf[pack_id * pack_size * j] = pack[j];
            thread_max = max(thread_max, pack[j]);
        }
    因此，在实现中，对Shared memory采用了新的内存布局，避免了同一个Warp访问相同bank的不同地址，避免了Bank Conflicts。
        #pragma unroll
        for (int j = 0; j < pack_size; ++j) {
            buf[num_packs * j + pack_id] = pack[j];
            thread_max = max(thread_max, pack[j]);
        }

    */
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* buf = reinterpret_cast<ComputeType*>(shared_buf); // (num_packs * VPT)
    const int num_packs = dim / VPT;
    int row_id = blockIdx.x;
    int tid = threadIdx.x;
    int cols_per_thread = (dim + TPB - 1) / TPB; 
    const T* row_x = x + row_id * dim;
    T* row_y = y + row_id * dim;

    // 1. Row max
    ComputeType thread_max = -Inf<ComputeType>();
    for (int pack_id = tid; pack_id < num_packs; pack_id += TPB) {
        ComputeType local_in[VPT];
        copy<sizeof(T) * VPT>(local_in, row_x + pack_id * VPT);
        #pragma unroll
        for (int i=0; i<VPT; ++i) {
            // buf[pack_id * VPT + i] = local_in[i]   // bank conflct, need optimize
            buf[num_packs * i + pack_id] = local_in[i];
            thread_max = max(thread_max, local_in[i]);
        }
    }
    const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, TPB>(thread_max);

    // 2. compute sum
    ComputeType thread_sum = 0;
    for (int pack_id=tid; pack_id<dim; pack_id += TPB) {
        #pragma unroll
        for (int i=0; i<VPT; ++i) {
            const ComputeType exp_x = exp(buf[num_packs * i + pack_id] - row_max);
            buf[num_packs * i + pack_id] = exp_x;
            thread_sum += exp_x;
        }
    }
    const ComputeType row_sum = BlockAllReduce(SumOp, ComputeType, TPB)(thread_sum);
    
    // 3. write back to gmem
    for (int pack_id = tid; pack_id < num_packs; pack_id += TPB) {
        ComputeType local_out[VPT];
        for (int i=0; i<VPT; ++i) {
            local_out[i] = buf[num_packs * i + pack_id] / row_sum;
        }
        copy<sizeof(ComputeType) * VPT, T>(row_y + pack_id * VPT, local_out);
    }
}


void test_warp_softamx() {
    constexpr int bs = 8, dim = 512;
    thrust::host_vector<float> h_x(bs * dim);
    for (int i=0; i<h_x.size(); ++i) {
        h_x[i] = (float)(rand() % 100);
    }
    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y(bs * dim, 0);

    float* d_x_ptr = thrust::raw_pointer_cast(d_x.data());
    float* d_y_ptr = thrust::raw_pointer_cast(d_y.data());
    
    constexpr int WARP_SIZE = 32;
    constexpr int VPT = 16 / sizeof(float);
    constexpr int WAPRS_PER_BLOCK = 128 / 32;

    constexpr int COLS_PER_THREAD = dim / WARP_SIZE;
    dim3 block(WARP_SIZE, WAPRS_PER_BLOCK);
    dim3 grid((bs + WAPRS_PER_BLOCK - 1) / WAPRS_PER_BLOCK);

    warp_softmax<WARP_SIZE, VPT, COLS_PER_THREAD><<<BPG, TPB>>>(d_x_ptr, dim, cols_per_thread, d_y_ptr);
}

void test_block_smem_softamx() {

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

    block_smem_softmax<TPB, VPT><<<BPG, TPB, shared_mem_bytes, 0>>>(d_x_ptr, dim, d_y_ptr);
}



int main(int argc, char const *argv[]) {
    
    constexpr int bs = 8, dim = 4096;
    thrust::host_vector<float> h_x(bs * dim);
    for (int i=0; i<h_x.size(); ++i) {
        h_x[i] = (float)(rand() % 100);
    }
    test_block_smem_softamx();
    test_warp_softamx();

    return 0;
}
