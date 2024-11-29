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

// // about CAS: https://blog.csdn.net/m0_52153904/article/details/130095643
// int atomicCAS(int* address, int compare, int val)
// {
//    int old = *address;
//    if(old == compare)
//        *address = val;
//    else
//        *address = old;
//    return(old);
// }

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


__device__ float2 operator+(float2 val1, float2 val2) {
  float2 ret;
  ret.x = val1.x + val2.x;
  ret.y = val1.y + val2.y;
  return ret;
}

  template<typename T>
  struct SumOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
  };

  template<typename T>
  struct MaxOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a, b); }
  };

  template<template<typename> class Op, typename T, int BLOCK_SIZE>
  __device__ T BlockAllReduce(T val) {
      __shared__ T shared[BLOCK_SIZE];
      int tid = threadIdx.x;
      shared[tid] = val;
      __syncthreads();
      #pragma unroll
      for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
          if (tid < offset) {
              shared[tid] = Op<T>()(shared[tid], shared[tid + offset]);
          }
          __syncthreads();
      }
      return shared[0];
  }

  template<template<typename> class Op, typename T, int WARP_WIDTH>
  __device__ T WarpAllReduce(T val) {
    #pragma unroll
    for (int mask = WARP_WIDTH / 2; mask > 0; mask >>= 1) {
      val += Op<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
  }


  // template<template<typename> class ReductionOp, typename T, int block_size>
  // __inline__ __device__ T BlockAllReduce(T val) {
  //   typedef cub::BlockReduce<T, block_size> BlockReduce;
  //   __shared__ typename BlockReduce::TempStorage temp_storage;
  //   __shared__ T result_broadcast;
  //   T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
  //   if (threadIdx.x == 0) { result_broadcast = result; }
  //   __syncthreads();
  //   return result_broadcast;
  // }

template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2>
{
    using type = uint16_t;
};
template <>
struct BytesToType<4>
{
    using type = uint32_t;
};
template <>
struct BytesToType<8>
{
    using type = uint64_t;
};
template <>
struct BytesToType<16>
{
    using type = float4;
};

template <int Bytes>
__device__ inline void copy(const void* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;

    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}
