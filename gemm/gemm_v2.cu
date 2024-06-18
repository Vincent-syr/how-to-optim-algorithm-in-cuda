#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <device_launch_parameters.h>
#include <iostream>

/*
thread block-level tiling优化
https://code.hitori.moe/post/cuda-gemm-fp32-optimization/#fn:2
https://zhuanlan.zhihu.com/p/478846788
https://github.com/tpoisonooo/how-to-optimize-gemm/blob/master/cuda/MMult_cuda_3.cu

naive 版每个 thread 都在做 global_mem -------> reg 的超远距离（473 cycle 延迟）搬运，
*/

/*
伪代码
    
for bk=0; bk<K; bk+=BLOCK
    float, sa[BLOCK][BLOCK]
    float sb[BLOCK][BLOCK]
*/

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

void cpuSgemm(
    float *a, float *b, float *c, const int M, const int N, const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}

bool checkSgemmResult(
    const float* y,
    const float* y_pred,
    const int M,
    const int N
) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float diff = y[OFFSET(m, n, N)] - y_pred[OFFSET(m, n, N)];
            if (abs(diff) > 1e-6) {
                printf("%d %d %f\n", m, n, diff);
                return false;
            }
        }
    }
    return true;
}

void print_matrix(int m, int n, float *a, int lda) {
  int i, j;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      printf("%.1f\t", OFFSET(i, j, lda));
    }
    printf("\n");
  }
  printf("\n");
}


template<uint32_t BLOCK>
__global__ void SgemmV2(
    float* __restrict__ a,
    float* __restrict__ b,
    float* __restrict__ c,
    const int M,
    const int N,
    const int K
) {
    /*
        BM = BN = BK = BLOCK, 特殊情况
    */

    __shared__ float sA[BLOCK][BLOCK];
    __shared__ float sB[BLOCK][BLOCK];
    float sum = 0.f;

    const int bm = blockIdx.y * BLOCK;
    const int bn = blockIdx.x * BLOCK;
    const int sm = threadIdx.y;
    const int sn = threadIdx.x;

    #pragma unroll
    for (int bk=0; bk<K; bk+=BLOCK) {
        const int gm = bm + sm;
        const int gn = bn + sn;
        sA[sm][sn] = a[OFFSET(gm, bk + sn, K)];
        sB[sm][sn] = b[OFFSET(bk + sm, gn, N)];
        __syncthreads();

        #pragma unroll
        for (int kk=0; kk < BLOCK; ++kk) {
            sum += sA[sm][kk] * sB[kk][sn];
        }
        __syncthreads();
    }
    c[OFFSET(bm + sm, bn + sn, N)] = sum;
}

template<int BM, int BN, int BK>
__global__ void SgemmV2_2(
    float* __restrict__ a,
    float* __restrict__ b,
    float* __restrict__ c,
    const int M,
    const int N,
    const int K
) {
    /*
        支持任意BM, BN, BK的代码
    */

    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];
    float sum = 0.f;

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int sm = threadIdx.y;
    const int sn = threadIdx.x;

    #pragma unroll
    for (int bk=0; bk<K; bk+=BK) {
        const int gm = bm + sm;
        const int gn = bn + sn;
        if (sn < BK) {
            sA[sm][sn] = a[OFFSET(gm, bk + sn, K)];
        }
        if (sm < BK) {
            sB[sm][sn] = b[OFFSET(bk + sm, gn, N)];
        }
        __syncthreads();

        #pragma unroll
        for (int kk=0; kk < BK; ++kk) {
            sum += sA[sm][kk] * sB[kk][sn];
        }
        __syncthreads();
    }
    c[OFFSET(bm + sm, bn + sn, N)] = sum;
}

// V2.3： 再增加一个float4对齐的



float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a = (float*)malloc(size_a);
    float *h_b = (float*)malloc(size_b);
    float *h_c = (float*)malloc(size_c);
    float* h_c_device = (float*)malloc(size_c);
    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    for (int i=0; i<M*K; ++i) {
        h_a[i] = (float)(rand() % 100);
    }
    for (int i=0; i<K*N; ++i) {
        h_b[i] = (float)(rand() % 100);
    }

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    // compute ground truth and check resutl
    cpuSgemm(h_a, h_b, h_c, M, N, K);
    cudaMemcpy(h_c_device, d_c, size_c, cudaMemcpyDeviceToHost);
    // check result
    bool is_right = checkSgemmResult(h_c_device, h_c, M, N);
    printf("Result: %s\n", is_right? "Passed" : "Errors");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return sec;
}


int main(int argc, char const *argv[])
{
    const int M = 512, N = 512, K = 512;
    // constexpr int BM = 32, BN = 32, BK=32;
    // constexpr int BLOCK = BM;
    constexpr int BM = 32, BN = 32, BK = 32; // blockDim最大1024，所以BM, BN最大32了，K小于BM和BK即可

    const int inner_repeat = 50;
    dim3 blockDim(BN, BM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    // void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) = SgemmV2<BLOCK>;
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) = SgemmV2_2<BM, BN, BK>;

    // warmup and check result
    constexpr int WARMUP_TIMES = 3;
    for (int i=0; i<WARMUP_TIMES; ++i) {
        testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
    }

    constexpr int TESTNUM = 2;
    constexpr int OUTER_REPEAT = 5;
    for (int i = 0; i< TESTNUM; ++i) {
        double total_sec = 0, min_sec = 1e6, max_sec = 0;

        for (int j=0; j<OUTER_REPEAT; ++j) {
            double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
            max_sec = std::max(max_sec, this_sec);
            min_sec = std::min(min_sec, this_sec);
            total_sec += this_sec;
        }
        double avg_sec = total_sec / OUTER_REPEAT;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;
        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);    // 3714.4789 Gflops
        // A100 FP32 理论算力： 19491.840 Gflops
        // A100 带宽：1555 GB/s

        // 达到了理论峰值的 1/5
    }
    return 0;
}
