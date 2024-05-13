#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

__global__
void PrintVec(const float* a, int size) {
    printf("==================================\n");
    for(int i=0; i<size; ++i) {
        printf("%f, ", a[i]);
    }
    printf("\n");
}

__global__
void PrintVec(const int32_t* a, int size) {
    printf("==================================\n");
    for(int i=0; i<size; ++i) {
        printf("%d, ", a[i]);
    }
    printf("\n");
}

//about CAS: https://blog.csdn.net/m0_52153904/article/details/130095643
//int atomicCAS(int* address, int compare, int val)
//{
//    old = *address;
//    if(old == compare)
//        *address = val;
//    else
//        *address = old;
//    return(old);
//}

// 封装好的atomicCAS不支持uint16_t, 所以需要将其转为int，下面fp32同理
__device__ 
double atomicAdd(uint16_t* address, uint16_t val) {
    // uint16_t old = *address;
    // uint16_t assumed;
    int* address_as_i = (int*)address;
    int old = *address;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, assumed + val);
    } while (assumed != old);
}

// 封装好的atmoicMax不支持fp32类型，所以我们这里需要针对fp32类型重载atomicMax
// fp32 type atomicMax from stackoverflow and nv developer forum: https://forums.developer.nvidia.com/t/cuda-atomicmax-for-float/194207
inline __device__ float atomicMax(float *address, float val) {
  int* address_as_i = (int*)address;
  int old = *address_as_i;
  int assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,  __float_as_int(fmaxf(val, __int_as_float(assumed))));

  } while (old != assumed);

  return __int_as_float(old);
}

inline __device__ float atomicMin(float *address, float val) {
  int* address_as_i = (int*)address;
  int old = *address_as_i;
  int assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,  __float_as_int(fminf(val, __int_as_float(assumed))));

  } while (old != assumed);

  return __int_as_float(old);
}