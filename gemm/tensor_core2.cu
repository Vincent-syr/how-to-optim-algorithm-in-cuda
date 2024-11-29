/*
https://github.com/BigNerd95/CUDASamples/blob/master/samples/0_Simple/cudaTensorCoreGemm/cudaTensorCoreGemm.cu#L385
*/


#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

// GPU configuration.

#define WARP_SIZE 32

#define M 16
#define N 16
#define K 16

// GEMM configuration.

#define M_TILES 256
#define N_TILES 256
#define K_TILES 256

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#define CHUNK_K 8

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)
// The macro below is used to shift rows of the A matrix and columns of the B matrix
// in shared memory to minimize possible bank conflicts.
// Before performing the nvcuda::wmma::mma_sync operation, the warp must load the matrix
// data using the nvcuda::wmma::load_matrix_sync operation. Although the memory access pattern
// is not specified for that function, each lane in the warp can read one or multiple matrix
// elements from different matrix rows or columns.
// For shared memory, such access can result in bank conflicts if different rows / columns
// of the matrix map to the same bank. By shifting each row and column by a few bytes, we
// make sure that they map to different banks, thus reducing the number of possible bank
// conflicts.
// The number of 8 two-byte "half" elements is chosen as the minimum possible shift because
// we must keep each row and column 128-bit aligned, as required by nvcuda::wmma::load_matrix_sync.
#define SKEW_HALF 8

#define SKEW_HALF 8

#define checkKernelErrors(expr) do {                                                        \
    expr;                                                                                   \
                                                                                            \
    cudaError_t __err = cudaGetLastError();                                                 \
    if (__err != cudaSuccess) {                                                             \
        printf("Line %d: '%s' failed: %s\n", __LINE__, # expr, cudaGetErrorString(__err));  \
        abort();                                                                            \
    }                                                                                       \
} while(0)

using namespace nvcuda;

__host__ void init_host_matrices(float *a, float *b, float *c)
{
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            a[i*K_GLOBAL+j] = (float)(rand() % 3);
        }
    }

    for (int i = 0; i < N_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            b[i*K_GLOBAL+j] = (float)(rand() % 3);
        }
    }

    for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
        c[t] = (float)(rand() % 3);
    }
}

__global__ void init_device_matrices(const float *A_h, const float *B_h, const float *C_h, half *A, half *B, float *C, float *D)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M_GLOBAL * K_GLOBAL; i += gridDim.x * blockDim.x)
        A[i] = __float2half(A_h[i]);

    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < N_GLOBAL * K_GLOBAL; i += gridDim.x * blockDim.x)
        B[i] = __float2half(B_h[i]);

    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M_GLOBAL * N_GLOBAL; i += gridDim.x * blockDim.x)
        C[i] = C_h[i];

    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M_GLOBAL * N_GLOBAL; i += gridDim.x * blockDim.x)
        D[i] = 0;
}

__global__ void compute_gemm(const half *A, const half *B, const float *C, float *D, float alpha, float beta)
{
    extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

    // Warp and lane identification.
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;

    // Offset in shared memory from which the B matrix is stored.
    const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

    // This pointer is used to access the C and D matrix tiles this warp computes.
    float *shmem_warp_tile_ptr = (float*)&shmem[0][0] + (warpId/2) * SHMEM_STRIDE * K * 2 + (warpId%2) * SHMEM_OFFSET;

    // This pointer is used to stream the C and D matrices block-wide tile to and from shared memory.
    float *shmem_warp_stream_ptr = (float*)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

    // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
    // each tile computation. Technically this is not generally correct (may result
    // in a loss of precision). Zero still needs to be specially handled though.
    beta /= alpha;

    // Each CTA slides along the 128 x 128 tiles from the top left corner of the matrix to the
    // right and down, and selects the next tile to compute. Once there's no such tile,
    // all warps in this CTA exit.
}


int main(int argc, char const *argv[])
{
    printf("Initializing...\n");

    int dev = findCudaDevice(argc, (const char **)argv);

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    // Tensor cores require a GPU of Volta (SM7X) architecture or higher.
    if (deviceProp.major < 7) {
        printf("cudaTensorCoreGemm requires requires SM 7.0 or higher to use Tensor Cores.  Exiting...\n");
        exit(EXIT_WAIVED);
    }

    printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
    printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
    printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

    float *A_h = NULL;
    float *B_h = NULL;
    float *C_h = NULL;

    checkCudaErrors(cudaMallocManaged((void**)&A_h, sizeof(float) * M_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMallocManaged((void**)&B_h, sizeof(float) * K_GLOBAL * N_GLOBAL));
    checkCudaErrors(cudaMallocManaged((void**)&C_h, sizeof(float) * M_GLOBAL * N_GLOBAL));

    half *A = NULL;
    half *B = NULL;
    float *C = NULL;
    float *D = NULL;

    checkCudaErrors(cudaMalloc((void**)&A, sizeof(half) * M_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc((void**)&B, sizeof(half) * N_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc((void**)&C, sizeof(float) * M_GLOBAL * N_GLOBAL));
    checkCudaErrors(cudaMalloc((void**)&D, sizeof(float) * M_GLOBAL * N_GLOBAL));

    assert(((unsigned long long)A) % 128 == 0);
    assert(((unsigned long long)B) % 128 == 0);
    assert(((unsigned long long)C) % 128 == 0);
    assert(((unsigned long long)D) % 128 == 0);

    init_host_matrices(A_h, B_h, C_h);

    printf("Preparing data for GPU...\n");

    checkKernelErrors((init_device_matrices<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK>>>(A_h, B_h, C_h, A, B, C, D)));

    checkCudaErrors(cudaDeviceSynchronize());

    enum { SHMEM_SZ = sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2 };
    printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);
    checkCudaErrors(cudaFuncSetAttribute(compute_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));

    printf("Computing...\n");

    cudaEvent_t start, stop;

    checkCudaErrors(cudaEventCreate(&start));    
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    const float alpha = 1.1f;
    const float beta = 1.2f;

    checkKernelErrors((compute_gemm<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ>>>(A, B, C, D, alpha, beta)));

    return 0;
}
