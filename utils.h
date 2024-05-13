#include <vector>
#include <iostream>
#include <stdio.h>

#include <cuda_runtime_api.h>


template<typename T>
static void PrintVec(const std::vector<T> vec) {
    for(T i : vec) {
        std::cerr << i << ", ";
    }
    std::cerr << std::endl;
}

template <typename T> void print_matrix(const int &m, const int &n, const T *A, const int &lda);

template <> void print_matrix(const int &m, const int &n, const float *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <> void print_matrix(const int &m, const int &n, const double *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}


// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// // cublas API error checking
// #define CUBLAS_CHECK(err)                                                                          \
//     do {                                                                                           \
//         cublasStatus_t err_ = (err);                                                               \
//         if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
//             std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
//             throw std::runtime_error("cublas error");                                              \
//         }                                                                                          \
//     } while (0)

