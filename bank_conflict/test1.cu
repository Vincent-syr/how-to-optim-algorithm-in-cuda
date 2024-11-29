
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>

using namespace std;

#include<stdio.h>
#include<time.h>
#define WARPSIZE 32
__global__ void kernel1(float* A) {
    __shared__ float data[32][32];
    int tid = threadIdx.x;
    int col = tid/WARPSIZE;
    int row = tid%WARPSIZE;
    data[row][col] = 100.f;
    A[tid] = data[row][col];
}


__global__ void kernel2(float* A) {
    __shared__ float data[32][32];
    int tid = threadIdx.x;
    int row = tid/WARPSIZE;
    int col = tid%WARPSIZE;
    data[row][col] = 100.f;
    A[tid] = data[row][col];
}

int main() {
    int blocksize = 32*32;
    float* h_A = (float*)malloc(sizeof(float)*blocksize);
    float* d_A;
    cudaMalloc(&d_A, sizeof(float)*blocksize);
 
    kernel1<<<1, blocksize>>>(d_A);
    cudaDeviceSynchronize();
    cudaMemcpy(h_A, d_A, blocksize*sizeof(float), cudaMemcpyDeviceToHost);

    kernel2<<<1, blocksize>>>(d_A);
    cudaDeviceSynchronize();
    cudaMemcpy(h_A, d_A, blocksize*sizeof(float), cudaMemcpyDeviceToHost);    

    cudaFree(d_A);
    free(h_A);
    return 0;
}
