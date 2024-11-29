# include "common.h"
# include "type.h"
# include "qfunc.h"
# include "layout.h"

# include "cute/tensor.hpp"
# include <cub/cub.cuh>


/**
 * Skip Rmsnorm 与 动态量化 的融合算子
 * 这个算子会执行 Add, Rmsnorm, Dynamic Quant 三个操作
 * Dynamic Quant 是指这个算子会在运行时统计输入的 per token min-max 并以此对输入进行量化
 * 使用公式 int8 value = clip(round(fp16 value / scale), -127, 127)
 *     其中 scale = abs(max(fp16 value)) / 127
 * 这个算子会执行 per token 量化，每一个 token 都将拥有一个属于它的 scale
 *
 * Skip Rmsnorm 会先执行 y1 = x + skip
 *              而后执行 y2 = rmsnorm(y1)
 * 返回 y1, y2, y2_scale 作为输出
 */
template <int VPT, int TPB, int32_t normalize_shape>
__global__
void _SkipRmsNormForward_fp16_i8(
  const fp16_t *x,                // 输入，形如 [batch, hidden dim]
  const fp16_t *weight,           // rmsnorm 的权重，形如 [hidden dim]
  const fp16_t *skip,             // 求和项，这是 skip rmsnorm 中的另一个输入，形如 [batch, hidden dim]
  const fp32_t eps,               // 1e-7
  fp16_t *o1,                     // x + skip
  fp32_t *o2_scale,               // quant(rmsnorm(x + skip))
  int8_t *o2                      // quant(rmsnorm(x + skip))
) {
    static_assert(normalize_shape % (TPB * VPT) == 0);
    static_assert(normalize_shape >= TPB * VPT);
    constexpr int32_t ITER = normalize_shape / TPB / VPT;

    int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    fp16_t inLocal[VPT * ITER]; fp16_t weightLocal[VPT * ITER];

# pragma unroll
    for(auto i = 0; i < ITER; i++) {
        copy<sizeof(fp16_t) * VPT>(&x[idx + i * TPB * VPT], &inLocal[i * VPT]);
        copy<sizeof(fp16_t) * VPT>(&skip[idx + i * TPB * VPT], &weightLocal[i * VPT]);
    }
    fp32_t accumulator = 0.0f; // accumulator
    fp32_t local_max   = eps;  // for int8 quant
    constexpr fp32_t r_normalize_shape = 1 / (float)(normalize_shape);

// step 1. compute x + skip
#pragma unroll
    for (auto i = 0; i < ITER * VPT; i++) {
        inLocal[i] = inLocal[i] + weightLocal[i];
    }

    for(auto i = 0; i < ITER; i++) {
        copy<sizeof(fp16_t) * VPT>(&inLocal[i * VPT], &o1[idx + i * TPB * VPT]);
        copy<sizeof(fp16_t) * VPT>(&weight[threadIdx.x * VPT + i * TPB * VPT], &weightLocal[i * VPT]);
    }

#pragma unroll
    for (auto i = 0; i < ITER * VPT; i++){
        auto _x = __half2float(inLocal[i]);
        auto _w = __half2float(weightLocal[i]);

        accumulator = accumulator + _x * _x;
        local_max   = max(local_max, abs(_x * _w));
    }

    using BlockReduce = cub::BlockReduce<fp32_t, TPB>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ fp32_t r_reduced;
    __shared__ fp32_t r_scale;

    const fp32_t global_max = BlockReduce(
        tempStorage).Reduce(local_max, cub::Max());
    __syncthreads();
    const fp32_t reduced = BlockReduce(tempStorage).Reduce(
        accumulator, cub::Sum()) * r_normalize_shape;

    if (threadIdx.x == 0) {
        r_reduced = rsqrt(reduced + eps);
        const fp32_t scale = min_max_range_to_scale(global_max * r_reduced, ATEX_INT8_LEVEL);
        o2_scale[blockIdx.x] = scale;
        r_scale = 1 / scale;
    }
    __syncthreads();

    int8_t outLocal[VPT * ITER];
    #pragma unroll
    for (auto i = 0; i < VPT * ITER; i++){
        fp32_t fp32_value = __half2float(inLocal[i]) * __half2float(weightLocal[i]) * r_reduced;
        outLocal[i] = quant_scalar<fp32_t, fp32_t, true>(
            fp32_value, r_scale, ATEX_INT8_MIN, ATEX_INT8_MAX);
    }

    for(auto i = 0; i < ITER; i++) {
        copy<sizeof(int8_t) * VPT>(&outLocal[i * VPT], &o2[idx + i * VPT * TPB]);
    }
}


/**
 * Skip Rmsnorm 与 动态量化 的融合算子
 * 这个算子会执行 Add, Rmsnorm, Dynamic Quant 三个操作
 * Dynamic Quant 是指这个算子会在运行时统计输入的 per token min-max 并以此对输入进行量化
 * 使用公式 int8 value = clip(round(fp16 value / scale), -127, 127)
 *     其中 scale = abs(max(fp16 value)) / 127
 * 这个算子会执行 per token 量化，每一个 token 都将拥有一个属于它的 scale
 *
 * Skip Rmsnorm 会先执行 y1 = x + skip
 *              而后执行 y2 = rmsnorm(y1)
 * 返回 y1, y2, y2_scale 作为输出
 */
std::tuple<Tensor, Tensor, Tensor> SkipRmsNormForward_fp16_i8(
    const Tensor x,               // 输入，形如 [batch, hidden dim]
    const Tensor weight,          // rmsnorm 的权重，形如 [hidden dim]
    const Tensor skip,            // 残差项，这是 skip rmsnorm 中的另一个输入，形如 [batch, hidden dim]
    const fp32_t eps              // 1e-7
) {
    TORCH_CHECK(x.is_cuda(), "Input Tensor x must be a Cuda Tensor.");
    TORCH_CHECK(weight.is_cuda(), "Input Tensor weight must be a Cuda Tensor.");
    TORCH_CHECK(skip.is_cuda(), "Input Tensor skip must be a Cuda Tensor.");

    TORCH_CHECK(x.scalar_type() == c10::ScalarType::Half, "Input Tensor x must be a FP16 Tensor.");
    TORCH_CHECK(weight.scalar_type() == c10::ScalarType::Half, "Input Tensor weight must be a FP16 Tensor.");
    TORCH_CHECK(skip.scalar_type() == c10::ScalarType::Half, "Input Tensor skip must be a FP16 Tensor.");

    const int32_t hidden_dim = x.size(-1);
    constexpr int32_t VPT    = 16 / sizeof(fp16_t);
    const int32_t grid_size  = x.numel() / hidden_dim;

    Tensor normalized_output = torch::empty(
        x.sizes(), torch::TensorOptions().dtype(torch::kChar).device(x.device())
    );
    Tensor output_quant_scale = torch::empty(
        {grid_size}, torch::TensorOptions().dtype(torch::kFloat).device(x.device())
    );
    Tensor skip_output = at::empty_like(x);

    switch(hidden_dim){
        case 768:
            _SkipRmsNormForward_fp16_i8<VPT, 768 / VPT, 768>
            <<<grid_size, 768 / VPT, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<fp16_t>(x), PTR<fp16_t>(weight), PTR<fp16_t>(skip), eps,
                PTR<fp16_t>(skip_output), PTR<fp32_t>(output_quant_scale), 
                PTR<int8_t>(normalized_output)
            );
            break;
        case 1024:
            _SkipRmsNormForward_fp16_i8<VPT, 1024 / VPT, 1024>
            <<<grid_size, 1024 / VPT, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<fp16_t>(x), PTR<fp16_t>(weight), PTR<fp16_t>(skip), eps,
                PTR<fp16_t>(skip_output), PTR<fp32_t>(output_quant_scale), 
                PTR<int8_t>(normalized_output)
            );
            break;
        case 1536:
            _SkipRmsNormForward_fp16_i8<VPT, 1536 / VPT, 1536>
            <<<grid_size, 1536 / VPT, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<fp16_t>(x), PTR<fp16_t>(weight), PTR<fp16_t>(skip), eps,
                PTR<fp16_t>(skip_output), PTR<fp32_t>(output_quant_scale), 
                PTR<int8_t>(normalized_output)
            );
            break;
        case 2048:
            _SkipRmsNormForward_fp16_i8<VPT, 2048 / VPT, 2048>
            <<<grid_size, 2048 / VPT, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<fp16_t>(x), PTR<fp16_t>(weight), PTR<fp16_t>(skip), eps,
                PTR<fp16_t>(skip_output), PTR<fp32_t>(output_quant_scale), 
                PTR<int8_t>(normalized_output)
            );
            break;
        case 4096:
            _SkipRmsNormForward_fp16_i8<VPT, 4096 / VPT, 4096>
            <<<grid_size, 4096 / VPT, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<fp16_t>(x), PTR<fp16_t>(weight), PTR<fp16_t>(skip), eps,
                PTR<fp16_t>(skip_output), PTR<fp32_t>(output_quant_scale), 
                PTR<int8_t>(normalized_output)
            );
            break;
        case 8192:
            _SkipRmsNormForward_fp16_i8<VPT, 8192 / VPT, 8192>
            <<<grid_size, 8192 / VPT, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<fp16_t>(x), PTR<fp16_t>(weight), PTR<fp16_t>(skip), eps,
                PTR<fp16_t>(skip_output), PTR<fp32_t>(output_quant_scale), 
                PTR<int8_t>(normalized_output)
            );
            break;
        case 10240:
            _SkipRmsNormForward_fp16_i8<VPT, 320, 10240>
            <<<grid_size, 320, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<fp16_t>(x), PTR<fp16_t>(weight), PTR<fp16_t>(skip), eps,
                PTR<fp16_t>(skip_output), PTR<fp32_t>(output_quant_scale), 
                PTR<int8_t>(normalized_output)
            );
            break;
        default:
            throw InvalidValueException("Failed to invoke RmsNorm function, "
                "as it does not support the data shape you are currently passing in. "
                "Please modify the data shape or modify the definition code in the rmsnorm.cu file.");
            break;
    }
    return {normalized_output, output_quant_scale, skip_output};
}
