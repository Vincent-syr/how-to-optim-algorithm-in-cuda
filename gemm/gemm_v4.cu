
/*
主旨: 
通过对A进行转置，解决v3中的SMEM的bank conflict问题


*/


#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <device_launch_parameters.h>
#include <iostream>


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
__global__ void SgemmV4(
    float* __restrict__ a,
    float* __restrict__ b,
    float* __restrict__ c,
    const int M,
    const int N,
    const int K
) {
    constexpr int BLOCK_M = BM, BLOCK_N = BN, BLOCK_K = BK;
    constexpr int BLOCK_SIZE = BM / TM;
    constexpr int BLOCK_M_COMPUTE = TM, BLOCK_N_COMPUTE = TN;
    constexpr int  shared_memory_A = BM * BK; 
    constexpr int shared_memory_B = BN * BLOCK_K;
    // s_a + sb维护的矩阵元素数量
    constexpr int shared_memory_element = shared_memory_A + shared_memory_B;
    constexpr int shared_memory_size = shared_memory_element * sizeof(float); // shared memory to use.
    constexpr float alpha = 1, beta = 0;

    // 该thread所属的block计算出的结果矩阵中的第一个元素，在C矩阵N方向上的偏移量
    // 如图例，对于(1,2)这个block，baseX = 1*16*8 = 128
    const size_t baseX = blockIdx.x * blockDim.x * BLOCK_N_COMPUTE;
    // 该thread所属的block计算出的结果矩阵中的第一个元素，在C矩阵M方向上的偏移量
    // 如图例，对于(1,2)这个block，baseX = 2*16*8 = 256
    const size_t baseY = blockIdx.y * blockDim.y * BLOCK_M_COMPUTE;

    const int moveNum = shared_memory_element / (BLOCK_SIZE * BLOCK_SIZE) / 2;
    // 该thread的tid，如图例，(2,1)这个thread的tid = 18
    const size_t baseIdx = threadIdx.y * blockDim.x + threadIdx.x;
    
    // 每个block中维护的线程数量
    constexpr size_t threadsNum = BLOCK_SIZE * BLOCK_SIZE;
    
    // 存放计算结果
    float resC[BLOCK_M_COMPUTE * BLOCK_N_COMPUTE] = {};

    // 在SMEM上开辟空间存放高亮红块subA, 高亮黄块subB(也就是前面说的s_a, s_b)
    __shared__ float subA[BLOCK_M * BLOCK_K];
    __shared__ float subB[BLOCK_N * BLOCK_K];
    
    // 在寄存器中，为渐变红regA和渐变黄regB开辟了存放空间
    float4 regB[BLOCK_M_COMPUTE / 4]; // hopefully, these should reside in register.
    float4 regA[BLOCK_M_COMPUTE / 4];

    // 该thread所属的block，要取的浅红色块的第一个元素，在矩阵A上的地址
    const float *baseA = a + baseY * K;
    // 该thread所属的block，要取的浅黄色块的第一个元素，在矩阵B上的地址
    const float *baseB = b + baseX;
    // N * 2^3
    const auto ldb8 = N << 3;   // K=8

    /*
    当前thread负责从global memory加载一部分高亮红、一部分高亮黄到SMEM，
    因此所有thread一起加载了完整的高亮红(s_a，本代码中也称为subA), 高亮黄(s_b, 即subB)到SMEM
    加载方式和上例中代码描述的一致，这里不再重复说明
    
    rowA: 该thread负责加载的第一个数在s_a中的row
    colA: 该thread负责加载的第一个数在s_a中的col
    rowB：该thread负责加载的第一个数在s_b中的row
    colB: 该thread负责加载的第一个数在s_b中的col
    */

    const int rowA = baseIdx >> 1, rowB = baseIdx >> 5, colA = (baseIdx & 1) << 2, colB = (baseIdx << 2) & 127;

    /*
    baseIdx即tid
    warpId：当前thread所属的warp id。这里0～31为warp0，32～63为warp1，以此类推。例如tid=18的
            线程属于warp0
    warpBaseId：即tid%32，即当前thread在所属warp中的相对位置，例如tid=18的线程在warp中的相对位置
                是18。tid = 33的线程在warp中的相对位置是1
    */
    int warpId = baseIdx >> 5, warpBaseId = baseIdx & 31;

    /*
    当前thread计算的(Tm, Tn)区域的第一个元素在其所属的block所维护的那块C矩阵中的位置
    例如当前thread的tid = 18，则rowC = 16, colC = 32
    */
    int rowC = ((warpId >> 1 << 2) + (warpBaseId & 3)) << 3, colC = (((warpId & 1) << 3) + (warpBaseId >> 2)) << 3;

    
    /*
    该thread计算的(Tm, Tn)区域的第一个元素，对应在完整的C矩阵中的地址
    */
    float *baseC = c + (baseY + rowC) * N + baseX + colC;


    for (int i=0; i<K; i+=BK) {
        /*
        1. 在block的单次循环中，需要把对应的s_a（高亮红）和s_b(高亮黄)从global memory
        加载到SMEM上，因此每个thread负责加载一部分s_a, s_b的数据，最后的__syncthreads()
        是保证thread们在正式计算前，都干完了自己加载的活，即完整的s_a, s_b已被加载到SMEM上
        */
        
        // // 加载当前thread所负责加载的s_a上的那4个数
        // // 这里是从global memory加载，所以计算的是在A矩阵上的位置
        // regA[0] = FLOAT4(baseA[OFFSET(rowA, colA, K)]);
        // // 加载当前thread所负责加载的s_b上的那4个数
        // regB[0] = FLOAT4(baseB[OFFSET(rowB, colB, N)]);

        // 加载当前thread所负责加载的s_a上的那4个数
        // 这里是从global memory加载，所以计算的是在A矩阵上的位置
        regA[0] = *reinterpret_cast<const float4 *>(baseA + rowA * K + colA);
        // 加载当前thread所负责加载的s_b上的那4个数
        regB[0] = *reinterpret_cast<const float4 *>(baseB + rowB * N + colB);


        // 对s_b正常装载4个数
        *reinterpret_cast<float4 *>(&subB[baseIdx * 4]) = regB[0];

        // 对s_a则做了转置，这是为了避免SMEM bank conflict
        subA[rowA + colA * BLOCK_M] = regA[0].x;
        subA[(rowA) + (colA + 1) * BLOCK_M] = regA[0].y;
        subA[(rowA) + (colA + 2) * BLOCK_M] = regA[0].z;
        subA[(rowA) + (colA + 3) * BLOCK_M] = regA[0].w;

        baseA += BK;
        baseB += ldb8;

        // 在所有thread间做一次同步，保证在下面的计算开始时，s_a, s_b相关的数据已经全部从global memory搬运到SMEM上了
        __syncthreads();

        #pragma unroll
        for (int ii = 0; ii < BK; ii++) {
            // 取出当前thread所要取的第一个float4渐变黄块 （32）
            regB[0] = *reinterpret_cast<float4 *>(&subB[colC + BLOCK_N * ii]);
            // 取出当前thread所要取的第二个float4渐变黄块 （36）
            regB[1] = *reinterpret_cast<float4 *>(&subB[colC + 4 + BLOCK_N * ii]);

            // 取出当前thread所要取的第一个float4渐变红块 （16）
            regA[0] = *reinterpret_cast<float4 *>(&subA[rowC + ii * BLOCK_M]);
            // 取出当前thread所要取的第二个float4渐变黄块 （20）
            regA[1] = *reinterpret_cast<float4 *>(&subA[(rowC + 4) + ii * BLOCK_M]);

            #pragma unroll
            for (int cpi = 0; cpi < TM / 4; cpi++) {
                #pragma unroll
                for (int cpj = 0; cpj < TN / 4; cpj++) {
                    resC[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].x * regB[cpj].x;
                    resC[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].x * regB[cpj].y;
                    resC[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].x * regB[cpj].z;
                    resC[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].x * regB[cpj].w;

                    resC[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].y * regB[cpj].x;
                    resC[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].y * regB[cpj].y;
                    resC[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].y * regB[cpj].z;
                    resC[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].y * regB[cpj].w;

                    resC[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].z * regB[cpj].x;
                    resC[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].z * regB[cpj].y;
                    resC[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].z * regB[cpj].z;
                    resC[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].z * regB[cpj].w;

                    resC[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].w * regB[cpj].x;
                    resC[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].w * regB[cpj].y;
                    resC[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].w * regB[cpj].z;
                    resC[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].w * regB[cpj].w;
                }
            }
        }
        __syncthreads();
    }

    // write back to global memory
    #pragma unroll
    for (int i = 0; i < BLOCK_M_COMPUTE; i++) {
        #pragma unroll
        for (int j = 0; j < BLOCK_N_COMPUTE; j += 4) {
            *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[i * N + j]);
            regA[0].x = regA[0].x * beta + alpha * resC[i * BLOCK_M_COMPUTE + j];
            regA[0].y = regA[0].y * beta + alpha * resC[i * BLOCK_M_COMPUTE + j + 1];
            regA[0].z = regA[0].z * beta + alpha * resC[i * BLOCK_M_COMPUTE + j + 2];
            regA[0].w = regA[0].w * beta + alpha * resC[i * BLOCK_M_COMPUTE + j + 3];
            *reinterpret_cast<float4 *>(&baseC[i * N + j]) = *reinterpret_cast<float4 *>(&regA[0]);
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

    // print_matrix(M, K, h_a, K);


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

int main(int argc, char const *argv[]) {
    constexpr int INNER_REPEAT = 10;

    /*
        smem方块
    */
    // const int M = 512, N = 512, K = 512;
    // constexpr int BM=32, BN=32, BK=32, TM=2, TN=2;
    // dim3 blockDim(BN / TN, BM / TM);
    // dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    // void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) = SgemmV3<BM, BN, BK, TM, TN>;

    /*
        此版本为教科书版本，sA[128, 8]，sB[8, 128], TM=TN=8，计算得到[128,128]子矩阵, 但性能相对更低，avg perf=1827.7675 Gflops
    */
    const int M = 512, N = 512, K = 512;
    constexpr int BM=128, BN=128, BK=8, TM=8, TN=8; // TK为了符合线程数要求，blockDim=16*16=256, 每个线程读float4读取，则一共256*4=1024个float，因此一个block里K=8
    // constexpr int BM=128, BN=128, BK=32, TM=4, TN=4;
    // constexpr int BM=32, BN=32, BK=32, TM=2, TN=2;
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) = SgemmV4<BM, BN, BK, TM, TN>;

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
        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);    //  5885.4771 Gflops
        // A100 FP32 理论算力： 19491.840 Gflops
        // A100 带宽：1555 GB/s

        // 达到了理论峰值的 30%
    }
    return 0;
}
