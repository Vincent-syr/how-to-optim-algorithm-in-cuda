#include <iostream>
#include <cuda_runtime.h>
using namespace std;

/*
ref: https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu
https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
*/
__global__ void transpose_baseline_kernel(
    float* __restrict__ idata, 
    const int M, 
    const int N, 
    float* __restrict__ odata
) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int src_stride = N, dst_stride = M;
    for (int i=gtid; i<M*N; i+=gridDim.x * blockDim.x) {
        int x = gtid / N, y = gtid % N;
        odata[y * dst_stride + x] = idata[x * src_stride + y];  // 读取连续（y），写入不连续
    }
}

void transpose() {
    int bs = 8, dim = 4096;
    constexpr int TPB = 256;
    const int grid = dim / TPB;
    transpose_baseline_kernel<<<grid, TPB>>>(x, y);
}


template<int TILE_DIM, int BLOCK_ROWS>
__global__ void transpose_smem_kernel(
    float* __restrict__ idata, 
    const int M, 
    const int N, 
    float* __restrict__ odata
) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;   // 即stride
    // 存在bank conflict
    for (int j=0; j<TILE_DIM; j+=BLOCK_ROWS) {  // 读取与写入连续
        tile[threadIdx.y + j][threadIdx.x] = idata[(y+j) * width + x];
    }
    __syncthreads();
    x = blockIdx.x * TILE_DIM + threadIdx.x;
    y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j=0; j<TILE_DIM; j+= BLOCK_ROWS) { // 写入gmem连续，读smem不连续，但由于smem带宽很高，所以无所谓
        odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

void transpose_smem() {
    const int nx = 1024;
    const int ny = 1024;
    
    constexpr int TILE_DIM = 32, BLOCK_ROWS = 8;
    dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    transpose_smem_kernel<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock>>>(x, M, N, y);
}


// avoid bank conflict
template<int TILE_DIM, int BLOCK_ROWS>
__global__ void transpose_no_back_conflict_kernel(
    float* __restrict__ idata, 
    const int M, 
    const int N, 
    float* __restrict__ odata
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;   // 即stride
    // copy to smem
    for (int j=0; j<TILE_DIM; j+=BLOCK_ROWS) {
        tile[threadIdx.y+j][threadIdx.x] = idata[(j+y)*width + x];
    }
    __syncthreads();
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j=0; j<TILE_DIM; j+=BLOCK_ROWS) {
        odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
    }

}

void transpose_smem() {
    const int nx = 1024;
    const int ny = 1024;
    
    constexpr int TILE_DIM = 32, BLOCK_ROWS = 8;
    dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    transpose_no_back_conflict_kernel<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock>>>(x, M, N, y);
}