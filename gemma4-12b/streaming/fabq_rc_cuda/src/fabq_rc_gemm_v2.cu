// fabq_rc_gemm_v2.cu - vectorized + tensor-core FABQ-RC GEMM kernels (v2).
//
// v2 of the inference kernels. Drop-in faster replacements for v1
// (fabq_rc_gemm.cu). Same numerical answer (within fp16 tolerance), same
// Python interface - just faster.
//
// What's here:
//   v2_int4_kernel      - vectorized scalar int4 GEMM. half2 loads on x,
//                         int8 scalar on W. The production path for
//                         single-token decode (B*T small).
//   v2_int4_via_fp16_tc_kernel
//                       - W4A16 tensor-core GEMM (int4 weight, fp16 act).
//                         The production path for batched eval/training
//                         (B*T >= 16). int8 W is dequantized to fp16 in
//                         shared memory on the fly, then fp16x fp16 WMMA.
//                         NOTE: not native int4 TC - that would require
//                         int4 activations, which explodes PPL at this bpw.
//   v2_binary_kernel    - vectorized binary-only GEMM with coalesced
//                         bit-byte unpacking. Each thread processes one
//                         bit-byte (8 weights) per iteration; adjacent
//                         threads load adjacent bytes for coalescing.
//   v2_mixed_kernel     - per-row int4-or-binary dispatch with the same
//                         vectorized paths. The general FABQ-RC case.
//   v2_embed_kernel     - quantized embedding lookup. Replaces the BF16
//                         embedding read called out as a v2 todo in
//                         build_streaming_notebook.py.
//
// Compile with -arch=sm_80 or higher for tensor cores; the scalar paths
// work on SM 7.5+.
//
// ============================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

namespace fabq_rc {

namespace wmma = nvcuda::wmma;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// Bit-unpack convention matches v1: bit_idx = row * in_features + i,
// byte_idx = bit_idx >> 3, bit_off = bit_idx & 7, LSB-first within byte.
__device__ __forceinline__ int unpack_bit_v2(
    const uint8_t* bits, int64_t row, int64_t in_features, int64_t i
) {
    int64_t bit_idx = row * in_features + i;
    int64_t byte_idx = bit_idx >> 3;
    int bit_off = bit_idx & 7;
    return ((bits[byte_idx] >> bit_off) & 1) ? 1 : -1;
}

// Block-wide float reduction. Caller provides smem of size (blockDim.x / 32).
__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFF, v, off);
    }
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    constexpr int NWARPS = 4;
    if (warp == 0) {
        v = (tid < NWARPS) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int off = NWARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xFFFFFFFF, v, off);
        }
    }
    return v;  // valid only on thread 0
}

// ===========================================================================
// V2_INT4 - vectorized scalar int4 GEMM (decode-friendly)
// ===========================================================================
//
// One block per (out_features_chunk, batch_token). Threads in a block stride
// over in_features with vectorized __half2 loads on x.
//
// Grid:  (ceil(out_features / BLOCK_O), B_T)
// Block: 128 threads

template <int BLOCK_THREADS, bool FUSE_BIAS>
__global__ void v2_int4_kernel(
    const __half* __restrict__ x,            // [B*T, in_features]
    const int8_t* __restrict__ int4_w,      // [n_int4, in_features]
    const __half* __restrict__ int4_scales, // [n_int4]
    const int64_t* __restrict__ row_to_int4,// [out_features] -> int4 idx, or -1
    const __half* __restrict__ bias,        // [out_features] or nullptr
    int B_T, int out_features, int in_features,
    __half* __restrict__ y                  // [B*T, out_features]
) {
    int tid = threadIdx.x;
    int o = blockIdx.x * BLOCK_THREADS + tid;
    int bt = blockIdx.y;
    if (o >= out_features) return;

    int64_t int4_row = row_to_int4[o];
    if (int4_row < 0) return;  // not an int4 row - skip (caller should dispatch elsewhere)

    float scale_f = __half2float(int4_scales[int4_row]);
    const int8_t* w_row = int4_w + int4_row * in_features;
    const __half* x_row = x + bt * in_features;

    int vec_count = in_features / 2;
    const __half2* x_vec = reinterpret_cast<const __half2*>(x_row);

    float acc = 0.0f;
    for (int p = 0; p < vec_count; p++) {
        __half2 xv2 = x_vec[p];
        int8_t w0 = w_row[2 * p];
        int8_t w1 = w_row[2 * p + 1];
        float x0 = __half2float(__low2half(xv2));
        float x1 = __half2float(__high2half(xv2));
        acc += x0 * (float(w0) * scale_f);
        acc += x1 * (float(w1) * scale_f);
    }
    if ((in_features & 1)) {
        int i_tail = in_features - 1;
        acc += __half2float(x_row[i_tail]) * (float(w_row[i_tail]) * scale_f);
    }

    if (FUSE_BIAS && bias != nullptr) {
        acc += __half2float(bias[o]);
    }
    y[bt * out_features + o] = __float2half(acc);
}

// ===========================================================================
// V2_BINARY - vectorized binary-only GEMM with coalesced bit access
// ===========================================================================
//
// One block per (out_features_row, batch_token). Adjacent threads handle
// adjacent bytes of binary_bits so loads are coalesced 32-bit transactions.
//
// Grid:  (out_features, B_T)
// Block: 128 threads

template <int BLOCK_THREADS, bool FUSE_BIAS>
__global__ void v2_binary_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ binary_bits,
    const __half* __restrict__ binary_scales,
    const uint8_t* __restrict__ codebook_idx,
    const __half* __restrict__ codebook,
    const int64_t* __restrict__ row_to_binary,
    const __half* __restrict__ bias,
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
    const uint8_t* bits_row      = binary_bits + bin_row * in_features;

    float acc = 0.0f;

    for (int blk = 0; blk < n_blocks; blk++) {
        int blk_start = blk * blocksize;
        int blk_end   = min(blk_start + blocksize, in_features);
        int blk_len   = blk_end - blk_start;

        float scale_f = __half2float(bin_scales_row[blk]);
        int cb_id     = (int)cb_idx_row[blk];
        const __half* cb = codebook + cb_id * max_blocksize;

        // 8 elements per thread iteration (one bit-byte). Adjacent threads
        // -> adjacent bytes -> coalesced 32-bit loads.
        int bytes_in_blk = (blk_len + 7) / 8;
        const uint8_t* bits_blk = bits_row + blk_start;

        for (int b = tid; b < bytes_in_blk; b += BLOCK_THREADS) {
            uint8_t byte = bits_blk[b];
            int local_base = b * 8;
            #pragma unroll
            for (int bit = 0; bit < 8; bit++) {
                int local_i = local_base + bit;
                if (local_i >= blk_len) break;
                int sign = ((byte >> bit) & 1) ? 1 : -1;
                float w = float(sign) * scale_f + __half2float(cb[local_i]);
                float xv = __half2float(x_row[blk_start + local_i]);
                acc += xv * w;
            }
        }
    }

    if (FUSE_BIAS && bias != nullptr) {
        acc += __half2float(bias[o]);
    }
    y[bt * out_features + o] = __float2half(acc);
}

// ===========================================================================
// V2_MIXED - mixed int4 + binary per row (the general case)
// ===========================================================================
//
// Each thread handles one row of O. Inside the thread we dispatch to the
// vectorized int4 or binary path. This is the production path for layers
// with mixed allocation.
//
// Grid:  (ceil(out_features / BLOCK_O), B_T)
// Block: 128 threads

template <int BLOCK_THREADS, bool FUSE_BIAS>
__global__ void v2_mixed_kernel(
    const __half* __restrict__ x,
    const int8_t* __restrict__ int4_w,
    const __half* __restrict__ int4_scales,
    const uint8_t* __restrict__ binary_bits,
    const __half* __restrict__ binary_scales,
    const uint8_t* __restrict__ codebook_idx,
    const __half* __restrict__ codebook,
    const int64_t* __restrict__ row_to_int4,
    const int64_t* __restrict__ row_to_binary,
    const __half* __restrict__ bias,
    int B_T, int out_features, int in_features,
    int n_blocks, int blocksize, int n_clusters, int max_blocksize,
    __half* __restrict__ y
) {
    int tid = threadIdx.x;
    int o = blockIdx.x * BLOCK_THREADS + tid;
    int bt = blockIdx.y;
    if (o >= out_features) return;

    int64_t int4_row = row_to_int4[o];
    int64_t bin_row  = row_to_binary[o];

    const __half* x_row = x + bt * in_features;
    float acc = 0.0f;

    if (int4_row >= 0) {
        float scale_f = __half2float(int4_scales[int4_row]);
        const int8_t* w_row = int4_w + int4_row * in_features;
        int vec_count = in_features / 2;
        const __half2* x_vec = reinterpret_cast<const __half2*>(x_row);
        for (int p = 0; p < vec_count; p++) {
            __half2 xv2 = x_vec[p];
            int8_t w0 = w_row[2 * p];
            int8_t w1 = w_row[2 * p + 1];
            float x0 = __half2float(__low2half(xv2));
            float x1 = __half2float(__high2half(xv2));
            acc += x0 * (float(w0) * scale_f);
            acc += x1 * (float(w1) * scale_f);
        }
        if (in_features & 1) {
            int i_tail = in_features - 1;
            acc += __half2float(x_row[i_tail]) * (float(w_row[i_tail]) * scale_f);
        }
    } else if (bin_row >= 0) {
        const __half* bin_scales_row = binary_scales + bin_row * n_blocks;
        const uint8_t* cb_idx_row    = codebook_idx + bin_row * n_blocks;
        const uint8_t* bits_row      = binary_bits + bin_row * in_features;

        for (int blk = 0; blk < n_blocks; blk++) {
            int blk_start = blk * blocksize;
            int blk_end   = min(blk_start + blocksize, in_features);
            int blk_len   = blk_end - blk_start;

            float scale_f = __half2float(bin_scales_row[blk]);
            int cb_id     = (int)cb_idx_row[blk];
            const __half* cb = codebook + cb_id * max_blocksize;

            int bytes_in_blk = (blk_len + 7) / 8;
            const uint8_t* bits_blk = bits_row + blk_start;
            int row_base = blk_start;

            for (int b = tid; b < bytes_in_blk; b += BLOCK_THREADS) {
                uint8_t byte = bits_blk[b];
                int local_base = b * 8;
                #pragma unroll
                for (int bit = 0; bit < 8; bit++) {
                    int local_i = local_base + bit;
                    if (local_i >= blk_len) break;
                    int sign = ((byte >> bit) & 1) ? 1 : -1;
                    float w = float(sign) * scale_f + __half2float(cb[local_i]);
                    float xv = __half2float(x_row[row_base + local_i]);
                    acc += xv * w;
                }
            }
        }
    } else {
        return;  // No data for this row.
    }

    if (FUSE_BIAS && bias != nullptr) {
        acc += __half2float(bias[o]);
    }
    y[bt * out_features + o] = __float2half(acc);
}

// ===========================================================================
// V2_INT4_VIA_FP16_TC - W4A16 tensor-core GEMM (batched path, B*T >= 16)
// ===========================================================================
//
// int4 weights are dequantized to fp16 in shared memory; activations stay
// fp16; fp16 x fp16 WMMA m16n16k16 mma. This is W4A16 - the only viable
// precision regime for FABQ-RC at 1.21 bpw (W4A4 would explode PPL).
// ===========================================================================
//
// Tile: BM x BN x BK = 128 x 16 x 16. 4 warps arranged 4x1. Each warp
// produces 32 x 16 of output = 2 m-tiles x 1 n-tile (each 16x16), so
// 2 mma calls per K-step per warp.
//
// int4 W is dequantized to fp16 in shared memory on the fly. For rows
// that are NOT int4 (row_to_int4 < 0), the dequant produces zeros - those
// rows are handled by v2_mixed_kernel instead.
//
// Constraints:
//   - in_features % 16 == 0
//   - B*T % 16 == 0 (caller pads with zeros if needed)
//   - BM=128 so each block handles 128 rows of O
//   - out_features % BM == 0 (caller rounds up + masks)
//
// Grid:  (ceil(out_features / 128), ceil(B*T / 16))
// Block: 128 threads (4 warps, warp_m = w, warp_n = 0)

template <int BM, int BN, int BK, int NWARPS_M, int NWARPS_N>
__global__ void v2_int4_via_fp16_tc_kernel(
    const __half* __restrict__ x,            // [B*T, in_features]
    const int8_t* __restrict__ int4_w,      // [n_int4, in_features]
    const __half* __restrict__ int4_scales, // [n_int4]
    const int64_t* __restrict__ row_to_int4,// [out_features] -> int4 idx, or -1
    const __half* __restrict__ bias,        // [out_features] or nullptr
    int B_T, int out_features, int in_features,
    __half* __restrict__ y                  // [B*T, out_features]
) {
    constexpr int NWARPS = NWARPS_M * NWARPS_N;
    constexpr int WARP_M = BM / NWARPS_M;     // 32
    constexpr int WARP_N = BN / NWARPS_N;     // 16
    constexpr int M_TILES = WARP_M / 16;      // 2
    constexpr int N_TILES = WARP_N / 16;      // 1

    static_assert(BM % 16 == 0, "BM must be a multiple of 16");
    static_assert(BN % 16 == 0, "BN must be a multiple of 16");
    static_assert(BK == 16,    "BK = 16 (one WMMA k-step)");
    static_assert(WARP_M % 16 == 0, "WARP_M must be a multiple of 16");
    static_assert(WARP_N % 16 == 0, "WARP_N must be a multiple of 16");

    int o_tile = blockIdx.x;
    int n_tile = blockIdx.y;  // B*T tile index
    int o_base = o_tile * BM;
    int n_base = n_tile * BN;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int warp_m  = warp_id % NWARPS_M;       // for NWARPS_N=1 this is warp_id
    int warp_n  = warp_id / NWARPS_M;

    int warp_m_base = warp_m * WARP_M;      // row offset within block tile
    int warp_n_base = warp_n * WARP_N;      // col offset within block tile

    // Shared memory: A_smem is W tile (BM x BK fp16). B_smem is x tile
    // (BK x BN fp16). C_smem is the warp-staged fp32 accumulators. All
    // padded by +8 to dodge bank conflicts on the stride-16 inner dim.
    __shared__ __half smem_A[BM][BK + 8];
    __shared__ __half smem_B[BK][BN + 8];
    __shared__ float smem_C[BM][BN + 8];

    // Per-warp output accumulators (fp32). 2 m-tiles x 1 n-tile.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[M_TILES][N_TILES];
    #pragma unroll
    for (int i = 0; i < M_TILES; i++) {
        #pragma unroll
        for (int j = 0; j < N_TILES; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    int n_k_tiles = in_features / BK;

    // Unrolled cooperative loads. We don't use lambdas (which would require
    // --extended-lambda on nvcc); the code is inlined into the K-loop body.
    for (int k_tile = 0; k_tile < n_k_tiles; k_tile++) {
        int k_base = k_tile * BK;

        // ---- load_A_tile ----
        // 128 threads, 2048 fp16 elements (BM*BK = 128*16). One row per
        // thread (16 fp16 = 2 uint4 stores).
        {
            int row = tid;             // 0..127 -> 128 rows of BM
            int o = o_base + row;
            if (o >= out_features) {
                uint4 zero = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&smem_A[row][0]) = zero;
                *reinterpret_cast<uint4*>(&smem_A[row][8]) = zero;
            } else {
                int64_t int4_row = row_to_int4[o];
                if (int4_row < 0) {
                    uint4 zero = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(&smem_A[row][0]) = zero;
                    *reinterpret_cast<uint4*>(&smem_A[row][8]) = zero;
                } else {
                    float scale_f = __half2float(int4_scales[int4_row]);
                    const int8_t* w_row = int4_w + int4_row * in_features + k_base;
                    // Load 16 int8 = 1 uint4 reinterpret, dequantize to fp16,
                    // store as 2 uint4 chunks in smem.
                    uint4 packed = *reinterpret_cast<const uint4*>(w_row);
                    __half out0  = __float2half(float((int8_t)(packed.x & 0xFF)) * scale_f);
                    __half out1  = __float2half(float((int8_t)((packed.x >> 8) & 0xFF)) * scale_f);
                    __half out2  = __float2half(float((int8_t)((packed.x >> 16) & 0xFF)) * scale_f);
                    __half out3  = __float2half(float((int8_t)((packed.x >> 24) & 0xFF)) * scale_f);
                    __half out4  = __float2half(float((int8_t)(packed.y & 0xFF)) * scale_f);
                    __half out5  = __float2half(float((int8_t)((packed.y >> 8) & 0xFF)) * scale_f);
                    __half out6  = __float2half(float((int8_t)((packed.y >> 16) & 0xFF)) * scale_f);
                    __half out7  = __float2half(float((int8_t)((packed.y >> 24) & 0xFF)) * scale_f);
                    __half out8  = __float2half(float((int8_t)(packed.z & 0xFF)) * scale_f);
                    __half out9  = __float2half(float((int8_t)((packed.z >> 8) & 0xFF)) * scale_f);
                    __half out10 = __float2half(float((int8_t)((packed.z >> 16) & 0xFF)) * scale_f);
                    __half out11 = __float2half(float((int8_t)((packed.z >> 24) & 0xFF)) * scale_f);
                    __half out12 = __float2half(float((int8_t)(packed.w & 0xFF)) * scale_f);
                    __half out13 = __float2half(float((int8_t)((packed.w >> 8) & 0xFF)) * scale_f);
                    __half out14 = __float2half(float((int8_t)((packed.w >> 16) & 0xFF)) * scale_f);
                    __half out15 = __float2half(float((int8_t)((packed.w >> 24) & 0xFF)) * scale_f);
                    uint4 first  = make_uint4(
                        *reinterpret_cast<uint32_t*>(&out0),
                        *reinterpret_cast<uint32_t*>(&out2),
                        *reinterpret_cast<uint32_t*>(&out4),
                        *reinterpret_cast<uint32_t*>(&out6));
                    uint4 second = make_uint4(
                        *reinterpret_cast<uint32_t*>(&out8),
                        *reinterpret_cast<uint32_t*>(&out10),
                        *reinterpret_cast<uint32_t*>(&out12),
                        *reinterpret_cast<uint32_t*>(&out14));
                    *reinterpret_cast<uint4*>(&smem_A[row][0])  = first;
                    *reinterpret_cast<uint4*>(&smem_A[row][8])  = second;
                }
            }
        }

        // ---- load_B_tile ----
        // 128 threads, 256 fp16 (BK*BN = 16*16). 2 fp16 per thread.
        {
            int row = tid / 8;        // 0..15
            int col_pair = tid & 7;   // 0..7
            int x_col_0 = n_base + col_pair * 2 + 0;
            int x_col_1 = n_base + col_pair * 2 + 1;
            int x_row   = k_base + row;
            __half v0 = __float2half(0.0f);
            __half v1 = __float2half(0.0f);
            if (x_row < in_features) {
                if (x_col_0 < B_T) v0 = x[x_col_0 * in_features + x_row];
                if (x_col_1 < B_T) v1 = x[x_col_1 * in_features + x_row];
            }
            smem_B[row][col_pair * 2 + 0] = v0;
            smem_B[row][col_pair * 2 + 1] = v1;
        }

        __syncthreads();

        // Each warp does its M_TILES x N_TILES mma calls.
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
        #pragma unroll
        for (int mi = 0; mi < M_TILES; mi++) {
            wmma::load_matrix_sync(a_frag,
                &smem_A[warp_m_base + mi * 16][0], BK + 8);
            #pragma unroll
            for (int ni = 0; ni < N_TILES; ni++) {
                wmma::load_matrix_sync(b_frag,
                    &smem_B[0][warp_n_base + ni * 16], BN + 8);
                wmma::mma_sync(c_frag[mi][ni], a_frag, b_frag, c_frag[mi][ni]);
            }
        }
        __syncthreads();
    }

    // Write accumulators back to y.
    #pragma unroll
    for (int mi = 0; mi < M_TILES; mi++) {
        #pragma unroll
        for (int ni = 0; ni < N_TILES; ni++) {
            wmma::store_matrix_sync(
                &smem_C[warp_m_base + mi * 16][warp_n_base + ni * 16],
                c_frag[mi][ni], BN + 8, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Cooperative write to y. 128 threads, BM*BN = 128*16 = 2048 fp16 per
    // block. Each thread writes 16 fp16 (one full row of BN=16 cols, in
    // one row of O).
    {
        int o = o_base + tid;
        if (o < out_features) {
            #pragma unroll
            for (int j = 0; j < BN; j++) {
                int n = n_base + j;
                if (n < B_T) {
                    float v = smem_C[tid][j];
                    if (bias != nullptr) v += __half2float(bias[o]);
                    y[n * out_features + o] = __float2half(v);
                }
            }
        }
    }
}

// ===========================================================================
// V2_EMBED - quantized embedding lookup
// ===========================================================================
//
// Reconstructs each token's embedding vector from the same FABQ-RC
// components used by linear layers (int4 channels + binary bits +
// codebook indices + scales). One block per token; threads cooperate on
// the embed_dim direction.
//
// Grid:  (B_T,)
// Block: 128 threads

template <int BLOCK_THREADS>
__global__ void v2_embed_kernel(
    const int64_t* __restrict__ token_ids,    // [B*T]
    const int8_t* __restrict__ int4_w,       // [n_int4, embed_dim]
    const __half* __restrict__ int4_scales,
    const int64_t* __restrict__ embed_int4_idx, // [vocab_size] -> int4 row idx, or -1
    const uint8_t* __restrict__ binary_bits,
    const __half* __restrict__ binary_scales, // [n_binary, n_blocks]
    const uint8_t* __restrict__ codebook_idx,
    const __half* __restrict__ codebook,
    const int64_t* __restrict__ embed_bin_idx, // [vocab_size] -> binary row idx, or -1
    int B_T, int embed_dim,
    int n_blocks, int blocksize, int n_clusters, int max_blocksize,
    __half* __restrict__ y                   // [B*T, embed_dim]
) {
    int bt  = blockIdx.x;
    int tid = threadIdx.x;
    if (bt >= B_T) return;

    int64_t token = token_ids[bt];
    int64_t int4_row = embed_int4_idx[token];
    int64_t bin_row  = embed_bin_idx[token];

    __half* y_row = y + bt * embed_dim;

    if (int4_row >= 0) {
        float scale_f = __half2float(int4_scales[int4_row]);
        const int8_t* w_row = int4_w + int4_row * embed_dim;
        int vec_count = embed_dim / 2;
        const __half2* x_vec = reinterpret_cast<const __half2*>(w_row);
        __half2* y_vec = reinterpret_cast<__half2*>(y_row);
        for (int p = tid; p < vec_count; p += BLOCK_THREADS) {
            __half2 wv;
            int8_t w0 = w_row[2 * p];
            int8_t w1 = w_row[2 * p + 1];
            wv = __halves2half2(
                __float2half(float(w0) * scale_f),
                __float2half(float(w1) * scale_f));
            y_vec[p] = wv;
        }
        if ((embed_dim & 1) && tid == 0) {
            int i_tail = embed_dim - 1;
            y_row[i_tail] = __float2half(float(w_row[i_tail]) * scale_f);
        }
    } else if (bin_row >= 0) {
        const __half* bin_scales_row = binary_scales + bin_row * n_blocks;
        const uint8_t* cb_idx_row    = codebook_idx + bin_row * n_blocks;
        const uint8_t* bits_row      = binary_bits + bin_row * embed_dim;

        for (int blk = 0; blk < n_blocks; blk++) {
            int blk_start = blk * blocksize;
            int blk_end   = min(blk_start + blocksize, embed_dim);
            int blk_len   = blk_end - blk_start;

            float scale_f = __half2float(bin_scales_row[blk]);
            int cb_id     = (int)cb_idx_row[blk];
            const __half* cb = codebook + cb_id * max_blocksize;

            for (int i = tid; i < blk_len; i += BLOCK_THREADS) {
                int sign = unpack_bit_v2(bits_row, 0, embed_dim, blk_start + i);
                float w = float(sign) * scale_f + __half2float(cb[i]);
                y_row[blk_start + i] = __float2half(w);
            }
        }
    }
    // else: token has no embedding (shouldn't happen) - leave zeros (already allocated).
}

// ===========================================================================
// Python-facing launchers
// ===========================================================================

torch::Tensor v2_gemm_int4(
    torch::Tensor x,
    torch::Tensor int4_w,
    torch::Tensor int4_scales,
    torch::Tensor row_to_int4,
    c10::optional<torch::Tensor> bias_opt,
    torch::Tensor y
) {
    TORCH_CHECK(x.is_cuda() && int4_w.is_cuda() && y.is_cuda());
    TORCH_CHECK(x.scalar_type() == at::kHalf);
    TORCH_CHECK(x.dim() == 2);
    int B_T = x.size(0);
    int in_features = x.size(1);
    int out_features = y.size(1);

    // Dispatch to TC kernel if B*T is large enough to amortize the tile
    // setup cost (smem fill, mma fragment setup). For small B*T the
    // vectorized scalar path wins.
    if (B_T >= 16 && in_features % 16 == 0 && out_features % 128 == 0) {
        constexpr int BM = 128, BN = 16, BK = 16;
        constexpr int NWARPS_M = 4, NWARPS_N = 1;
        constexpr int BLOCK_THREADS = NWARPS_M * NWARPS_N * 32;
        dim3 grid((out_features + BM - 1) / BM, (B_T + BN - 1) / BN);
        dim3 block(BLOCK_THREADS);
        auto stream = at::cuda::getCurrentCUDAStream();

        const __half* bias_ptr = nullptr;
        if (bias_opt.has_value() && bias_opt->defined()) {
            bias_ptr = reinterpret_cast<const __half*>(bias_opt->data_ptr<at::Half>());
        }
        v2_int4_via_fp16_tc_kernel<BM, BN, BK, NWARPS_M, NWARPS_N>
            <<<grid, block, 0, stream.stream()>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            int4_w.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(int4_scales.data_ptr<at::Half>()),
            row_to_int4.data_ptr<int64_t>(),
            bias_ptr,
            B_T, out_features, in_features,
            reinterpret_cast<__half*>(y.data_ptr<at::Half>())
        );
        return y;
    }

    // Fallback: vectorized scalar for small B*T or odd shapes.
    constexpr int BLOCK = 128;
    dim3 grid((out_features + BLOCK - 1) / BLOCK, B_T);
    dim3 block(BLOCK);
    auto stream = at::cuda::getCurrentCUDAStream();

    const __half* bias_ptr = nullptr;
    bool fuse_bias = false;
    if (bias_opt.has_value() && bias_opt->defined()) {
        bias_ptr = reinterpret_cast<const __half*>(bias_opt->data_ptr<at::Half>());
        fuse_bias = true;
    }
    if (fuse_bias) {
        v2_int4_kernel<BLOCK, true><<<grid, block, 0, stream.stream()>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            int4_w.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(int4_scales.data_ptr<at::Half>()),
            row_to_int4.data_ptr<int64_t>(),
            bias_ptr,
            B_T, out_features, in_features,
            reinterpret_cast<__half*>(y.data_ptr<at::Half>())
        );
    } else {
        v2_int4_kernel<BLOCK, false><<<grid, block, 0, stream.stream()>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            int4_w.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(int4_scales.data_ptr<at::Half>()),
            row_to_int4.data_ptr<int64_t>(),
            nullptr,
            B_T, out_features, in_features,
            reinterpret_cast<__half*>(y.data_ptr<at::Half>())
        );
    }
    return y;
}

torch::Tensor v2_gemm_binary(
    torch::Tensor x,
    torch::Tensor binary_bits,
    torch::Tensor binary_scales,
    torch::Tensor codebook_idx,
    torch::Tensor codebook,
    torch::Tensor row_to_binary,
    c10::optional<torch::Tensor> bias_opt,
    torch::Tensor y,
    int64_t n_blocks, int64_t blocksize, int64_t n_clusters, int64_t max_blocksize
) {
    TORCH_CHECK(x.is_cuda() && binary_bits.is_cuda() && y.is_cuda());
    TORCH_CHECK(x.scalar_type() == at::kHalf);
    int B_T = x.size(0);
    int in_features = x.size(1);
    int out_features = y.size(1);

    constexpr int BLOCK = 128;
    dim3 grid(out_features, B_T);
    dim3 block(BLOCK);
    auto stream = at::cuda::getCurrentCUDAStream();

    const __half* bias_ptr = nullptr;
    bool fuse_bias = false;
    if (bias_opt.has_value() && bias_opt->defined()) {
        bias_ptr = reinterpret_cast<const __half*>(bias_opt->data_ptr<at::Half>());
        fuse_bias = true;
    }
    if (fuse_bias) {
        v2_binary_kernel<BLOCK, true><<<grid, block, 0, stream.stream()>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            binary_bits.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(binary_scales.data_ptr<at::Half>()),
            codebook_idx.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(codebook.data_ptr<at::Half>()),
            row_to_binary.data_ptr<int64_t>(),
            bias_ptr,
            B_T, out_features, in_features,
            (int)n_blocks, (int)blocksize, (int)n_clusters, (int)max_blocksize,
            reinterpret_cast<__half*>(y.data_ptr<at::Half>())
        );
    } else {
        v2_binary_kernel<BLOCK, false><<<grid, block, 0, stream.stream()>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            binary_bits.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(binary_scales.data_ptr<at::Half>()),
            codebook_idx.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(codebook.data_ptr<at::Half>()),
            row_to_binary.data_ptr<int64_t>(),
            nullptr,
            B_T, out_features, in_features,
            (int)n_blocks, (int)blocksize, (int)n_clusters, (int)max_blocksize,
            reinterpret_cast<__half*>(y.data_ptr<at::Half>())
        );
    }
    return y;
}

torch::Tensor v2_gemm_mixed(
    torch::Tensor x,
    torch::Tensor int4_w,
    torch::Tensor int4_scales,
    torch::Tensor binary_bits,
    torch::Tensor binary_scales,
    torch::Tensor codebook_idx,
    torch::Tensor codebook,
    torch::Tensor row_to_int4,
    torch::Tensor row_to_binary,
    c10::optional<torch::Tensor> bias_opt,
    torch::Tensor y,
    int64_t n_blocks, int64_t blocksize, int64_t n_clusters, int64_t max_blocksize
) {
    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(x.scalar_type() == at::kHalf);
    int B_T = x.size(0);
    int in_features = x.size(1);
    int out_features = y.size(1);

    constexpr int BLOCK = 128;
    dim3 grid((out_features + BLOCK - 1) / BLOCK, B_T);
    dim3 block(BLOCK);
    auto stream = at::cuda::getCurrentCUDAStream();

    const __half* bias_ptr = nullptr;
    bool fuse_bias = false;
    if (bias_opt.has_value() && bias_opt->defined()) {
        bias_ptr = reinterpret_cast<const __half*>(bias_opt->data_ptr<at::Half>());
        fuse_bias = true;
    }
    if (fuse_bias) {
        v2_mixed_kernel<BLOCK, true><<<grid, block, 0, stream.stream()>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            int4_w.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(int4_scales.data_ptr<at::Half>()),
            binary_bits.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(binary_scales.data_ptr<at::Half>()),
            codebook_idx.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(codebook.data_ptr<at::Half>()),
            row_to_int4.data_ptr<int64_t>(),
            row_to_binary.data_ptr<int64_t>(),
            bias_ptr,
            B_T, out_features, in_features,
            (int)n_blocks, (int)blocksize, (int)n_clusters, (int)max_blocksize,
            reinterpret_cast<__half*>(y.data_ptr<at::Half>())
        );
    } else {
        v2_mixed_kernel<BLOCK, false><<<grid, block, 0, stream.stream()>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            int4_w.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(int4_scales.data_ptr<at::Half>()),
            binary_bits.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(binary_scales.data_ptr<at::Half>()),
            codebook_idx.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(codebook.data_ptr<at::Half>()),
            row_to_int4.data_ptr<int64_t>(),
            row_to_binary.data_ptr<int64_t>(),
            nullptr,
            B_T, out_features, in_features,
            (int)n_blocks, (int)blocksize, (int)n_clusters, (int)max_blocksize,
            reinterpret_cast<__half*>(y.data_ptr<at::Half>())
        );
    }
    return y;
}

torch::Tensor v2_embed_lookup(
    torch::Tensor token_ids,
    torch::Tensor int4_w,
    torch::Tensor int4_scales,
    torch::Tensor embed_int4_idx,
    torch::Tensor binary_bits,
    torch::Tensor binary_scales,
    torch::Tensor codebook_idx,
    torch::Tensor codebook,
    torch::Tensor embed_bin_idx,
    torch::Tensor y,
    int64_t n_blocks, int64_t blocksize, int64_t n_clusters, int64_t max_blocksize
) {
    TORCH_CHECK(token_ids.is_cuda());
    TORCH_CHECK(token_ids.scalar_type() == at::kLong);
    TORCH_CHECK(token_ids.dim() == 1);
    int B_T = token_ids.size(0);
    int embed_dim = y.size(1);

    constexpr int BLOCK = 128;
    dim3 grid(B_T);
    dim3 block(BLOCK);
    auto stream = at::cuda::getCurrentCUDAStream();

    v2_embed_kernel<BLOCK><<<grid, block, 0, stream.stream()>>>(
        token_ids.data_ptr<int64_t>(),
        int4_w.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(int4_scales.data_ptr<at::Half>()),
        embed_int4_idx.data_ptr<int64_t>(),
        binary_bits.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(binary_scales.data_ptr<at::Half>()),
        codebook_idx.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(codebook.data_ptr<at::Half>()),
        embed_bin_idx.data_ptr<int64_t>(),
        B_T, embed_dim,
        (int)n_blocks, (int)blocksize, (int)n_clusters, (int)max_blocksize,
        reinterpret_cast<__half*>(y.data_ptr<at::Half>())
    );
    return y;
}

}  // namespace fabq_rc
