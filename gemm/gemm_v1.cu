#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <device_launch_parameters.h>
#include <iostream>

// naive gemm
/*
    每个线程管一个m或者n，K维度进行累加
*/

// https://zhuanlan.zhihu.com/p/518857175
// https://github.com/nicolaswilde/cuda-sgemm/blob/main/sgemm.cu

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

__global__ void naiveSgemm(
    float* __restrict__ a,
    float* __restrict__ b,
    float* __restrict__ c,
    const int M,
    const int N,
    const int K
) {
    // 全局线程索引，col major
	// int threadId = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

    int m = blockIdx.y * blockDim.y + threadIdx.y;  // 
    int n = blockIdx.x * blockDim.x + threadIdx.x;  // warp在此维度
    if (m < M && n < N) {
        float psum = 0.0;
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        }
        c[OFFSET(m, n, N)] = psum;
    }
}


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
    constexpr int BM = 32, BN = 32;
    const int inner_repeat = 50;
    dim3 blockDim(BN, BM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) = naiveSgemm;

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
        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);    // 2465.0523 Gflops
        // A100 FP32 理论算力： 19491.840 Gflops
        // A100 带宽：1555 GB/s

        // 达到了理论峰值的 1/12
    }
    return 0;
}




