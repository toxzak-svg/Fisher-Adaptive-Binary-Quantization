// fabq_rc_gemm.cu - the actual CUDA kernel for FABQ-RC native-quantized matmul.
//
// This is v1: scalar arithmetic, no tensor cores. It is correct and
// memory-safe but slow compared to cuBLAS. The structure is set up so v2 can
// drop in mma.sync / wmma calls for the int4 submatrix without changing the
// Python interface.
//
// Three public kernels:
//   1. fabq_rc_gemm_int4_only  - if a layer is 100% int4, fast path
//   2. fabq_rc_gemm_binary_only - if a layer is 100% binary, slow path
//   3. fabq_rc_gemm_mixed       - mixed int4 + binary per-row (the general case)
//
// The Python side dispatches based on which rows are which.
//
// Memory layout note: the kernel reads binary_bits as a packed uint8 array
// where bit (row * in_features + i) is at byte (row * in_features + i) / 8,
// bit offset (row * in_features + i) % 8, LSB-first within the byte. This
// matches the bit-packing in fabq_rc_quant.cpp.
//
// ============================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace fabq_rc {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ int unpack_bit(
    const uint8_t* bits, int64_t row, int64_t in_features, int64_t i
) {
    int64_t bit_idx = row * in_features + i;
    int64_t byte_idx = bit_idx >> 3;
    int bit_off = bit_idx & 7;
    return ((bits[byte_idx] >> bit_off) & 1) ? 1 : -1;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFF, v, off);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Int4-only kernel
// One block per (batch_token, output_channel). Each thread accumulates
// over a strided range of input features. Warp reduction at the end.
// ---------------------------------------------------------------------------
//
// Grid:  (out_features, B*T)
// Block: (256 threads)

template <int BLOCK_THREADS>
__global__ void fabq_rc_gemm_int4_kernel(
    const __half* __restrict__ x,           // [B*T, in_features]
    const int8_t* __restrict__ int4_w,     // [n_int4, in_features]
    const __half* __restrict__ int4_scales,// [n_int4]
    const int64_t* __restrict__ row_to_int4,// [out_features] -> int4 row idx, or -1
    int B_T, int out_features, int in_features,
    __half* __restrict__ y                 // [B*T, out_features]
) {
    int o = blockIdx.x;
    int bt = blockIdx.y;
    int tid = threadIdx.x;

    int64_t int4_row = row_to_int4[o];
    if (int4_row < 0) return;  // should not be called for this row

    float scale_f = __half2float(int4_scales[int4_row]);
    const int8_t* w_row = int4_w + int4_row * in_features;
    const __half* x_row = x + bt * in_features;

    float acc = 0.0f;
    for (int i = tid; i < in_features; i += BLOCK_THREADS) {
        float w = float(w_row[i]) * scale_f;
        float xv = __half2float(x_row[i]);
        acc += xv * w;
    }

    // Block reduction via shared mem
    __shared__ float smem[BLOCK_THREADS / 32];
    int lane = tid & 31;
    int warp = tid >> 5;
    acc = warp_reduce_sum(acc);
    if (lane == 0) smem[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (tid < BLOCK_THREADS / 32) ? smem[lane] : 0.0f;
        acc = warp_reduce_sum(acc);
        if (tid == 0) {
            y[bt * out_features + o] = __float2half(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// Binary-only kernel
// One block per (batch_token, output_channel). Each thread handles a strided
// range of blocks, then within each block accumulates over the input dim.
// ---------------------------------------------------------------------------
//
// Grid:  (out_features, B*T)
// Block: (128 threads)

template <int BLOCK_THREADS>
__global__ void fabq_rc_gemm_binary_kernel(
    const __half* __restrict__ x,           // [B*T, in_features]
    const uint8_t* __restrict__ binary_bits,// packed [n_binary * in_features / 8]
    const __half* __restrict__ binary_scales,// [n_binary, n_blocks]
    const uint8_t* __restrict__ codebook_idx,// [n_binary, n_blocks]
    const __half* __restrict__ codebook,    // [n_clusters, max_blocksize]
    const int64_t* __restrict__ row_to_binary,// [out_features] -> binary row idx, or -1
    int B_T, int out_features, int in_features,
    int n_blocks, int blocksize, int n_clusters, int max_blocksize,
    __half* __restrict__ y
) {
    int o = blockIdx.x;
    int bt = blockIdx.y;
    int tid = threadIdx.x;

    int64_t bin_row = row_to_binary[o];
    if (bin_row < 0) return;

    const __half* x_row = x + bt * in_features;
    const __half* bin_scales_row = binary_scales + bin_row * n_blocks;
    const uint8_t* cb_idx_row    = codebook_idx + bin_row * n_blocks;

    float acc = 0.0f;

    for (int blk = 0; blk < n_blocks; blk++) {
        int blk_start = blk * blocksize;
        int blk_end   = min(blk_start + blocksize, in_features);
        int blk_len   = blk_end - blk_start;

        __half scale_h = bin_scales_row[blk];
        float scale_f  = __half2float(scale_h);
        int cb_id      = (int)cb_idx_row[blk];
        const __half* cb = codebook + cb_id * max_blocksize;

        for (int local_i = tid; local_i < blk_len; local_i += BLOCK_THREADS) {
            int sign = unpack_bit(binary_bits, bin_row, in_features, blk_start + local_i);
            float w_from_bit = float(sign) * scale_f;
            float w_cb       = __half2float(cb[local_i]);
            float w          = w_from_bit + w_cb;
            float xv         = __half2float(x_row[blk_start + local_i]);
            acc += xv * w;
        }
    }

    __shared__ float smem[BLOCK_THREADS / 32];
    int lane = tid & 31;
    int warp = tid >> 5;
    acc = warp_reduce_sum(acc);
    if (lane == 0) smem[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (tid < BLOCK_THREADS / 32) ? smem[lane] : 0.0f;
        acc = warp_reduce_sum(acc);
        if (tid == 0) {
            y[bt * out_features + o] = __float2half(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// Mixed int4 + binary kernel
// One block per (batch_token, output_channel). The row's int4/binary
// designation is checked once; then the appropriate path runs.
// ---------------------------------------------------------------------------

template <int BLOCK_THREADS>
__global__ void fabq_rc_gemm_mixed_kernel(
    const __half* __restrict__ x,
    const int8_t* __restrict__ int4_w,
    const __half* __restrict__ int4_scales,
    const uint8_t* __restrict__ binary_bits,
    const __half* __restrict__ binary_scales,
    const uint8_t* __restrict__ codebook_idx,
    const __half* __restrict__ codebook,
    const int64_t* __restrict__ row_to_int4,
    const int64_t* __restrict__ row_to_binary,
    int B_T, int out_features, int in_features,
    int n_blocks, int blocksize, int n_clusters, int max_blocksize,
    __half* __restrict__ y
) {
    int o = blockIdx.x;
    int bt = blockIdx.y;
    int tid = threadIdx.x;

    int64_t int4_row = row_to_int4[o];
    int64_t bin_row  = row_to_binary[o];

    const __half* x_row = x + bt * in_features;
    float acc = 0.0f;

    if (int4_row >= 0) {
        float scale_f = __half2float(int4_scales[int4_row]);
        const int8_t* w_row = int4_w + int4_row * in_features;
        for (int i = tid; i < in_features; i += BLOCK_THREADS) {
            float w = float(w_row[i]) * scale_f;
            acc += __half2float(x_row[i]) * w;
        }
    } else if (bin_row >= 0) {
        const __half* bin_scales_row = binary_scales + bin_row * n_blocks;
        const uint8_t* cb_idx_row    = codebook_idx + bin_row * n_blocks;

        for (int blk = 0; blk < n_blocks; blk++) {
            int blk_start = blk * blocksize;
            int blk_end   = min(blk_start + blocksize, in_features);
            int blk_len   = blk_end - blk_start;
            float scale_f = __half2float(bin_scales_row[blk]);
            int cb_id     = (int)cb_idx_row[blk];
            const __half* cb = codebook + cb_id * max_blocksize;

            for (int local_i = tid; local_i < blk_len; local_i += BLOCK_THREADS) {
                int sign = unpack_bit(binary_bits, bin_row, in_features, blk_start + local_i);
                float w = float(sign) * scale_f + __half2float(cb[local_i]);
                acc += __half2float(x_row[blk_start + local_i]) * w;
            }
        }
    } else {
        // Row with no quantized data (shouldn't happen for active layers)
        return;
    }

    __shared__ float smem[BLOCK_THREADS / 32];
    int lane = tid & 31;
    int warp = tid >> 5;
    acc = warp_reduce_sum(acc);
    if (lane == 0) smem[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (tid < BLOCK_THREADS / 32) ? smem[lane] : 0.0f;
        acc = warp_reduce_sum(acc);
        if (tid == 0) {
            y[bt * out_features + o] = __float2half(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// Bias add (separate kernel, fused optionally in v2)
// ---------------------------------------------------------------------------

__global__ void fabq_rc_bias_add_kernel(
    __half* __restrict__ y,           // [B*T, out_features]
    const __half* __restrict__ bias,  // [out_features]
    int B_T, int out_features
) {
    int bt = blockIdx.x;
    int o  = blockIdx.y * blockDim.x + threadIdx.x;
    if (o >= out_features) return;
    y[bt * out_features + o] = __hadd(y[bt * out_features + o], bias[o]);
}

// ============================================================================
// Python-facing launchers
// ============================================================================

torch::Tensor fabq_rc_gemm_int4(
    torch::Tensor x,                  // [B*T, in_features] fp16, CUDA
    torch::Tensor int4_w,             // [n_int4, in_features] int8, CUDA
    torch::Tensor int4_scales,        // [n_int4] fp16, CUDA
    torch::Tensor row_to_int4,        // [out_features] int64, CUDA (or -1)
    torch::Tensor y                   // [B*T, out_features] fp16, CUDA (pre-allocated)
) {
    TORCH_CHECK(x.is_cuda() && int4_w.is_cuda() && y.is_cuda());
    TORCH_CHECK(x.scalar_type() == at::kHalf);
    TORCH_CHECK(x.dim() == 2);
    int B_T = x.size(0);
    int in_features = x.size(1);
    int out_features = y.size(1);

    dim3 grid(out_features, B_T);
    dim3 block(256);
    auto stream = at::cuda::getCurrentCUDAStream();
    fabq_rc_gemm_int4_kernel<256><<<grid, block, 0, stream.stream()>>>(
        reinterpret_cast<__half*>(x.data_ptr<at::Half>()),
        int4_w.data_ptr<int8_t>(),
        reinterpret_cast<__half*>(int4_scales.data_ptr<at::Half>()),
        row_to_int4.data_ptr<int64_t>(),
        B_T, out_features, in_features,
        reinterpret_cast<__half*>(y.data_ptr<at::Half>())
    );
    return y;
}

torch::Tensor fabq_rc_gemm_binary(
    torch::Tensor x,
    torch::Tensor binary_bits,
    torch::Tensor binary_scales,
    torch::Tensor codebook_idx,
    torch::Tensor codebook,
    torch::Tensor row_to_binary,
    torch::Tensor y,
    int64_t n_blocks, int64_t blocksize, int64_t n_clusters, int64_t max_blocksize
) {
    TORCH_CHECK(x.is_cuda() && binary_bits.is_cuda() && y.is_cuda());
    TORCH_CHECK(x.scalar_type() == at::kHalf);
    int B_T = x.size(0);
    int in_features = x.size(1);
    int out_features = y.size(1);

    dim3 grid(out_features, B_T);
    dim3 block(128);
    auto stream = at::cuda::getCurrentCUDAStream();
    fabq_rc_gemm_binary_kernel<128><<<grid, block, 0, stream.stream()>>>(
        reinterpret_cast<__half*>(x.data_ptr<at::Half>()),
        binary_bits.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(binary_scales.data_ptr<at::Half>()),
        codebook_idx.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(codebook.data_ptr<at::Half>()),
        row_to_binary.data_ptr<int64_t>(),
        B_T, out_features, in_features,
        (int)n_blocks, (int)blocksize, (int)n_clusters, (int)max_blocksize,
        reinterpret_cast<__half*>(y.data_ptr<at::Half>())
    );
    return y;
}

torch::Tensor fabq_rc_gemm_mixed(
    torch::Tensor x,
    torch::Tensor int4_w,
    torch::Tensor int4_scales,
    torch::Tensor binary_bits,
    torch::Tensor binary_scales,
    torch::Tensor codebook_idx,
    torch::Tensor codebook,
    torch::Tensor row_to_int4,
    torch::Tensor row_to_binary,
    torch::Tensor y,
    int64_t n_blocks, int64_t blocksize, int64_t n_clusters, int64_t max_blocksize
) {
    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(x.scalar_type() == at::kHalf);
    int B_T = x.size(0);
    int in_features = x.size(1);
    int out_features = y.size(1);

    dim3 grid(out_features, B_T);
    dim3 block(256);
    auto stream = at::cuda::getCurrentCUDAStream();
    fabq_rc_gemm_mixed_kernel<256><<<grid, block, 0, stream.stream()>>>(
        reinterpret_cast<__half*>(x.data_ptr<at::Half>()),
        int4_w.data_ptr<int8_t>(),
        reinterpret_cast<__half*>(int4_scales.data_ptr<at::Half>()),
        binary_bits.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(binary_scales.data_ptr<at::Half>()),
        codebook_idx.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(codebook.data_ptr<at::Half>()),
        row_to_int4.data_ptr<int64_t>(),
        row_to_binary.data_ptr<int64_t>(),
        B_T, out_features, in_features,
        (int)n_blocks, (int)blocksize, (int)n_clusters, (int)max_blocksize,
        reinterpret_cast<__half*>(y.data_ptr<at::Half>())
    );
    return y;
}

void fabq_rc_add_bias(torch::Tensor y, torch::Tensor bias) {
    TORCH_CHECK(y.is_cuda() && bias.is_cuda());
    TORCH_CHECK(y.scalar_type() == at::kHalf);
    int B_T = y.size(0);
    int out_features = y.size(1);
    dim3 grid(B_T);
    dim3 block(256);
    int n_blocks_x = (out_features + 256 - 1) / 256;
    dim3 grid_y(n_blocks_x);
    auto stream = at::cuda::getCurrentCUDAStream();
    fabq_rc_bias_add_kernel<<<grid, dim3(256), 0, stream.stream()>>>(
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(bias.data_ptr<at::Half>()),
        B_T, out_features
    );
}

}  // namespace fabq_rc
