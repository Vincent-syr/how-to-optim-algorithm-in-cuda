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

https://github.com/nicolaswilde/cuda-sgemm/blob/main/sgemm.cu
https://mp.weixin.qq.com/s/9zpgkIrfBuJnR8UDNCHCZQ
*/

/*
伪代码
    
for bk=0; bk<K; bk+=BLOCK
    float, sa[BLOCK][BLOCK]
    float sb[BLOCK][BLOCK]

*/

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

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
      printf("%.1f\t", a[OFFSET(i, j, lda)]);
    }
    printf("\n");
  }
  printf("\n");
}

__device__ void print_matrix_device(int m, int n, float *a, int lda) {
  int i, j;
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      printf("%.1f\t", a[OFFSET(i, j, lda)]);
    }
    printf("\n");
  }
  printf("\n");
}


template<int BM, int BN, int BK, int TM, int TN>
__global__ void SgemmV3(
    float* __restrict__ a,
    float* __restrict__ b,
    float* __restrict__ c,
    const int M,
    const int N,
    const int K
) {
    constexpr int NUM_PER_THREAD = TM;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    float r_c[TM][TN] = {0.0};

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int sm = threadIdx.y * NUM_PER_THREAD;
    const int sn = threadIdx.x * NUM_PER_THREAD;

    #pragma unroll
    for (int bk=0; bk < K; bk += BK) {
        // load to shared mem
        #pragma unroll
        for (int i = 0; i < TM ; ++i) {
            const int gm = bm + sm + i;
            for (int j=0; j< TN; ++j) {
                const int gn = bn + sn + j;
                if (sn + j < BK) {
                    sA[sm + i][sn + j] = a[OFFSET(gm, bk + sn + j, K)];
                }
                if (sm + i < BK) {
                    sB[sm + i][sn + j] = b[OFFSET(bk + sm + i, gn, N)];
                }
            }

        }

        __syncthreads();

        if (tid == 0 && bk == 0) {
            // print_matrix_device(BM, BK, (float*)sA, BK);
        }

        #pragma unroll
        for (int k=0; k < BK; ++k) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    int sm_m = sm + m;
                    int sm_n = sn + n;
                    r_c[m][n] += sA[sm_m][k] * sB[k][sm_n];
                }
            }
        }
        __syncthreads();
    }
    // write to global mem
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            const int gm = bm + sm + i;
            const int gn = bn + sn + j;
            c[OFFSET(gm, gn, N)] = r_c[i][j];
        }
    }
}

__device__ int log2(int k) {
    int power = 0;
    while(k > 1) {
        k = k >> 1;
        ++power;
    }
    return power;
}

template<int BM, int BN, int BK, int TM, int TN>
__global__ void SgemmV3_2(
    float* __restrict__ a,
    float* __restrict__ b,
    float* __restrict__ c,
    const int M,
    const int N,
    const int K
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x; // thread在对应block内的行id
    const int ty = threadIdx.y; // thread在对应block内的列id
    const int tid = ty * blockDim.x + tx; // thread在对应block中的全局id（从左到右，从上到下，从0开始逐一标）

    /*
    在SMEM上对A和B，分别开辟大小为(BM, BK), (BK, BN)的空间
    对应到图例中，s_a为高亮红，s_b为高亮黄
    */
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

   /*
    初始化当前thread所维护的C矩阵（确定长度的数组，应该是定义在寄存器上的）
    */
    float r_c[TM][TN] = {0.0};

    /*
    示例：
    对于tid = 0的thread，以下四个值分别为((0, 0), (0, 0)),意味着它负责把s_a(0,0)开始的连续4个数，s_b(0,0)开始的连续4个数，从global memory加载到SMEM

    对于tid = 1的thread，以下四个值分别为((0, 4), (0, 4)),
    意味着它负责把s_a(0,4)开始的连续4个数，s_b(0,4)开始的连续4个数，从global memory加载到SMEM

    对于tid = 2的thread，以下四个值分别为((1, 0), (0, 8))
    此时s_a第一行的8个数已经被前面的thread取完了，所以现在从s_a第二行开始取，s_b第一行没取完，继续进行

    对于tid = 18的thread，以下四个值分别为((9, 0), (0, 72))，含义同上
    */

    // 当前thread负责把A中的相关数据从global memory加载到SMEM，
    // 这里在计算该thread负责加载的第一个数在s_a中的row

    int power_k = log2(BK / 4);
    int load_a_smem_m = tid >> (power_k);
    int load_a_smem_k = (tid & (power_k)) << 2;  // (tid % 2 == 0) ? 0 : 4, col of s_a
    // int load_a_smem_m = tid >> 1;  // tid/2, row of s_a
    // int load_a_smem_k = (tid & 1) << 2;  // (tid % 2 == 0) ? 0 : 4, col of s_a

    // 当前thread负责把B中的相关数据从global memory加载到SMEM，
    // 这里在计算该thread负责加载的第一个数在s_b中的row

    int load_b_smem_k = tid >> 5;   // tid/32, row of s_b
    int load_b_smem_n = (tid & 31) << 2;    // (tid % 32) * 4, col of s_b

    int load_a_gmem_m = by * BM + load_a_smem_m;    // global row of a
    int load_b_gmem_n = bx * BN + load_b_smem_n;     // global col of b
    
    #pragma unroll
    for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
        // load to smem
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[OFFSET(load_a_gmem_m, load_a_gmem_k, K)]);

        int load_b_gmem_k = bk * BK + load_b_smem_k;
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[OFFSET(load_b_gmem_k, load_b_gmem_n, N)]);
        __syncthreads();

        // 计算环节， 每个线程load元素
        #pragma unroll
        for (int k=0; k<BK; ++k) {
            #pragma unroll
            for (int m = 0; m<TM; ++m) {
                #pragma unroll
                for (int n=0; n<TN; ++n) {
                    int comp_a_smem_m = ty * TM + m;
                    int comp_b_smem_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }
        // 做一次同步，保证所有的thread都计算完当前所维护的（TM, TN）块
         __syncthreads();
    }
    /*
    3. 
    此时，所有的block已做完循环，
    我们把当前thread计算出的结果（存放在r_c中，尺寸为(Tm, Tn)）写回
    global memory上的C矩阵对应位置中
    */

    for (int i=0; i<TM; ++i) {
        int c_gemm_m = by * BM + ty * TM + i;
        for (int j=0; j<TN; j+=4) {
            int c_gemm_n = bx * BN + tx * TN + j;
            // 将这4个数以FLOAT4写回global memory
            FLOAT4(c[OFFSET(c_gemm_m, c_gemm_n, N)]) = FLOAT4(r_c[i][j]);
        }
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
    // printf("h_c cpu result: \n");
    // print_matrix(M, N, h_c, N);
    // printf("h_c device result: \n");
    // print_matrix(M, N, h_c_device, N);

    bool is_right = checkSgemmResult(h_c_device, h_c, M, N);
    printf("Result: %s\n", is_right? "Passed" : "Errors");
    if (!is_right) {
        exit(0);
    }

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

    constexpr int INNER_REPEAT = 50;
    constexpr int BM=128, BN=128, BK=8, TM=8, TN=8; // TK为了符合线程数要求，blockDim=16*16=256, 每个线程读float4读取，则一共256*4=1024个float，因此一个block里K=8
    // constexpr int BM=128, BN=128, BK=32, TM=4, TN=4;

    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) = SgemmV3_2<BM, BN, BK, TM, TN>;

    // warmup and check result
    constexpr int WARMUP_TIMES = 3;
    for (int i=0; i<WARMUP_TIMES; ++i) {
        testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, INNER_REPEAT);
    }

    constexpr int TESTNUM = 2;
    constexpr int OUTER_REPEAT = 5;
    for (int i = 0; i< TESTNUM; ++i) {
        double total_sec = 0, min_sec = 1e6, max_sec = 0;

        for (int j=0; j<OUTER_REPEAT; ++j) {
            double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, INNER_REPEAT);
            max_sec = std::max(max_sec, this_sec);
            min_sec = std::min(min_sec, this_sec);
            total_sec += this_sec;
        }
        double avg_sec = total_sec / OUTER_REPEAT;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;
        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);    // 1848.4789 Gflops
        // A100 FP32 理论算力： 19491.840 Gflops
        // A100 带宽：1555 GB/s

        // 达到了理论峰值的 1/10
    }
    return 0;
}
