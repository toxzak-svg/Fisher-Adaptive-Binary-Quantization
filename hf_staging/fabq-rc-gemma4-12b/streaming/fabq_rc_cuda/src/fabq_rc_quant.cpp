// fabq_rc_quant.cpp - CPU-side quantization helpers (write to / read from disk).
//
// This is the C++ implementation of:
//   - BF16 weights -> FABQ-RC packed format (one layer at a time)
//   - FABQ-RC packed format -> disk .bin file
//   - disk .bin file -> in-memory torch tensors
//
// The Python side calls into these for the bucket-build path. The streaming
// notebook path uses these to (optionally) re-quantize from BF16 if the
// pre-quantized shards are missing.

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "fabq_rc_format.h"

namespace fabq_rc {

// ---------------------------------------------------------------------------
// Quantize a single FP16/BF16 weight matrix to FABQ-RC components.
// Pure CPU implementation. The kernel is what runs at inference; this just
// packs the bytes for the kernel to read.
// ---------------------------------------------------------------------------
//
// Inputs:
//   weight:        [out_features, in_features] float32 (caller converts from bf16)
//   int4_channels: [n_int4] int64 (indices into out_features dim)
//   binary_channels: [n_binary] int64
//   blocksize:     int (e.g. 128)
//   codebook:      [n_clusters, max_blocksize] float32, the shared k-means codebook
//
// Outputs (all newly-allocated):
//   int4_weights:    [n_int4, in_features] int8
//   int4_scales:     [n_int4] float16
//   binary_bits:     packed uint8 array, ceil(n_binary * in_features / 8) bytes
//   binary_scales:   [n_binary, n_blocks] float16
//   codebook_idx:    [n_binary, n_blocks] uint8

std::vector<torch::Tensor> quantize_weight_matrix(
    torch::Tensor weight,            // [out, in] float32
    torch::Tensor int4_channels,     // [n_int4] int64
    torch::Tensor binary_channels,   // [n_binary] int64
    int64_t blocksize,
    torch::Tensor codebook           // [n_clusters, max_blocksize] float32
) {
    TORCH_CHECK(weight.scalar_type() == at::kFloat);
    TORCH_CHECK(weight.dim() == 2);
    TORCH_CHECK(weight.is_contiguous());
    int64_t out_features = weight.size(0);
    int64_t in_features  = weight.size(1);
    int64_t n_int4       = int4_channels.size(0);
    int64_t n_binary     = binary_channels.size(0);
    int64_t n_clusters   = codebook.size(0);
    int64_t max_blocksize = codebook.size(1);

    auto float_opts = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    auto half_opts  = at::TensorOptions().dtype(at::kHalf).device(at::kCPU);
    auto int8_opts  = at::TensorOptions().dtype(at::kChar).device(at::kCPU);
    auto int64_opts = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
    auto uint8_opts = at::TensorOptions().dtype(at::kByte).device(at::kCPU);

    // ---- INT4 channels: simple symmetric per-row quantization to int8 range ----
    auto int4_weights = at::empty({n_int4, in_features}, int8_opts);
    auto int4_scales  = at::empty({n_int4}, half_opts);
    const float* w_data = weight.data_ptr<float>();
    const int64_t* int4_ch = int4_channels.data_ptr<int64_t>();

    for (int64_t k = 0; k < n_int4; k++) {
        int64_t row = int4_ch[k];
        const float* row_ptr = w_data + row * in_features;
        float max_abs = 0.0f;
        for (int64_t i = 0; i < in_features; i++) {
            float v = std::abs(row_ptr[i]);
            if (v > max_abs) max_abs = v;
        }
        float scale = (max_abs > 1e-12f) ? (max_abs / 127.0f) : 1.0f;
        int8_t* out_w = int4_weights.data_ptr<int8_t>() + k * in_features;
        for (int64_t i = 0; i < in_features; i++) {
            float q = std::round(row_ptr[i] / scale);
            if (q >  127.0f) q =  127.0f;
            if (q < -127.0f) q = -127.0f;
            out_w[i] = (int8_t)q;
        }
        int4_scales.data_ptr<at::Half>()[k] = (at::Half)scale;
    }

    // ---- BINARY channels: bit-pack + per-block scale + nearest codebook entry ----
    int64_t n_blocks = (in_features + blocksize - 1) / blocksize;
    int64_t bits_bytes = (n_binary * in_features + 7) / 8;

    auto binary_bits   = at::zeros({bits_bytes}, uint8_opts);
    auto binary_scales = at::empty({n_binary, n_blocks}, half_opts);
    auto codebook_idx  = at::empty({n_binary, n_blocks}, uint8_opts);

    const int64_t* bin_ch = binary_channels.data_ptr<int64_t>();
    const float* cb_data  = codebook.data_ptr<float>();

    std::vector<float> residual_buf(blocksize);

    for (int64_t k = 0; k < n_binary; k++) {
        int64_t row = bin_ch[k];
        const float* row_ptr = w_data + row * in_features;

        for (int64_t blk = 0; blk < n_blocks; blk++) {
            int64_t blk_start = blk * blocksize;
            int64_t blk_end   = std::min(blk_start + blocksize, in_features);
            int64_t blk_len   = blk_end - blk_start;

            // Per-block std as scale (matches the kernel's `binary_scale` interpretation)
            float sum = 0.0f, sum_sq = 0.0f;
            for (int64_t i = 0; i < blk_len; i++) {
                float v = row_ptr[blk_start + i];
                sum += v; sum_sq += v * v;
            }
            float mean = sum / (float)blk_len;
            float var  = sum_sq / (float)blk_len - mean * mean;
            float std  = std::sqrt(std::max(var, 0.0f)) + 1e-8f;
            binary_scales.data_ptr<at::Half>()[k * n_blocks + blk] = (at::Half)std;

            // Binarize to +/- std
            for (int64_t i = 0; i < blk_len; i++) {
                int sign = (row_ptr[blk_start + i] > 0.0f) ? 1 : 0;
                int64_t bit_idx = k * in_features + blk_start + i;
                int64_t byte_idx = bit_idx >> 3;
                int bit_off      = bit_idx & 7;
                if (sign) {
                    binary_bits.data_ptr<uint8_t>()[byte_idx] |= (uint8_t)(1u << bit_off);
                }
            }

            // Compute residual = W - bit_recon, find nearest codebook entry
            for (int64_t i = 0; i < blk_len; i++) {
                float w = row_ptr[blk_start + i];
                int sign = (w > 0.0f) ? 1 : -1;
                residual_buf[i] = w - sign * std;
            }
            // Zero-pad the residual to max_blocksize for the L2 distance
            std::vector<float> padded(max_blocksize, 0.0f);
            std::memcpy(padded.data(), residual_buf.data(), blk_len * sizeof(float));

            int best_c = 0;
            float best_d = std::numeric_limits<float>::infinity();
            for (int64_t c = 0; c < n_clusters; c++) {
                float d = 0.0f;
                const float* cb = cb_data + c * max_blocksize;
                for (int64_t i = 0; i < max_blocksize; i++) {
                    float diff = padded[i] - cb[i];
                    d += diff * diff;
                }
                if (d < best_d) { best_d = d; best_c = (int)c; }
            }
            codebook_idx.data_ptr<uint8_t>()[k * n_blocks + blk] = (uint8_t)best_c;
        }
    }

    return {int4_weights, int4_scales, binary_bits, binary_scales, codebook_idx};
}

// ---------------------------------------------------------------------------
// Write a single layer's quantized buffers to a .bin file
// ---------------------------------------------------------------------------

void write_layer_to_file(
    std::string path,
    int64_t layer_index,
    int64_t in_features, int64_t out_features,
    torch::Tensor int4_channels,
    torch::Tensor int4_weights,
    torch::Tensor int4_scales,
    torch::Tensor binary_channels,
    torch::Tensor binary_bits,
    torch::Tensor binary_scales,
    torch::Tensor codebook_idx,
    int64_t blocksize,
    c10::optional<torch::Tensor> bias
) {
    std::ofstream f(path, std::ios::binary);
    TORCH_CHECK(f.is_open(), "Cannot open file: ", path);

    int64_t n_int4   = int4_channels.size(0);
    int64_t n_binary = binary_channels.size(0);
    int64_t n_blocks = binary_scales.size(1);

    LayerHeader hdr{};
    hdr.magic        = kLayerMagic;
    hdr.version      = kFormatVersion;
    hdr.layer_index  = (uint32_t)layer_index;
    hdr.reserved     = 0;
    hdr.in_features  = (uint32_t)in_features;
    hdr.out_features = (uint32_t)out_features;
    hdr.n_int4       = (uint32_t)n_int4;
    hdr.n_binary     = (uint32_t)n_binary;
    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

    f.write(reinterpret_cast<const char*>(int4_channels.data_ptr<int64_t>()),
            n_int4 * sizeof(int64_t));
    f.write(reinterpret_cast<const char*>(int4_weights.data_ptr<int8_t>()),
            n_int4 * in_features * sizeof(int8_t));
    f.write(reinterpret_cast<const char*>(int4_scales.data_ptr<at::Half>()),
            n_int4 * sizeof(at::Half));

    f.write(reinterpret_cast<const char*>(binary_channels.data_ptr<int64_t>()),
            n_binary * sizeof(int64_t));
    int64_t bits_bytes = (n_binary * in_features + 7) / 8;
    f.write(reinterpret_cast<const char*>(binary_bits.data_ptr<uint8_t>()),
            bits_bytes);
    uint32_t n_blocks_u32 = (uint32_t)n_blocks;
    f.write(reinterpret_cast<const char*>(&n_blocks_u32), sizeof(uint32_t));
    uint32_t blocksize_u32 = (uint32_t)blocksize;
    f.write(reinterpret_cast<const char*>(&blocksize_u32), sizeof(uint32_t));
    f.write(reinterpret_cast<const char*>(binary_scales.data_ptr<at::Half>()),
            n_binary * n_blocks * sizeof(at::Half));
    f.write(reinterpret_cast<const char*>(codebook_idx.data_ptr<uint8_t>()),
            n_binary * n_blocks * sizeof(uint8_t));

    if (bias.has_value() && bias->defined()) {
        f.write(reinterpret_cast<const char*>(bias->data_ptr<at::Half>()),
                out_features * sizeof(at::Half));
    } else {
        // NaN sentinel
        std::vector<uint16_t> nan_buf(out_features, kFp16NaN);
        f.write(reinterpret_cast<const char*>(nan_buf.data()),
                out_features * sizeof(uint16_t));
    }
}

// ---------------------------------------------------------------------------
// Read a layer's quantized buffers from a .bin file
// ---------------------------------------------------------------------------

struct LoadedLayer {
    int64_t layer_index;
    int64_t in_features;
    int64_t out_features;
    int64_t n_int4;
    int64_t n_binary;
    int64_t n_blocks;
    int64_t blocksize;
    torch::Tensor int4_channels;
    torch::Tensor int4_weights;
    torch::Tensor int4_scales;
    torch::Tensor binary_channels;
    torch::Tensor binary_bits;
    torch::Tensor binary_scales;
    torch::Tensor codebook_idx;
    c10::optional<torch::Tensor> bias;
};

LoadedLayer read_layer_from_file(std::string path) {
    std::ifstream f(path, std::ios::binary);
    TORCH_CHECK(f.is_open(), "Cannot open file: ", path);

    LayerHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    TORCH_CHECK(hdr.magic == kLayerMagic, "Bad magic in ", path);
    TORCH_CHECK(hdr.version == kFormatVersion, "Bad version in ", path);

    int64_t in_features  = hdr.in_features;
    int64_t out_features = hdr.out_features;
    int64_t n_int4       = hdr.n_int4;
    int64_t n_binary     = hdr.n_binary;

    auto int64_opts = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
    auto half_opts  = at::TensorOptions().dtype(at::kHalf).device(at::kCPU);
    auto int8_opts  = at::TensorOptions().dtype(at::kChar).device(at::kCPU);
    auto uint8_opts = at::TensorOptions().dtype(at::kByte).device(at::kCPU);

    auto int4_channels = at::empty({n_int4}, int64_opts);
    auto int4_weights  = at::empty({n_int4, in_features}, int8_opts);
    auto int4_scales   = at::empty({n_int4}, half_opts);
    f.read(reinterpret_cast<char*>(int4_channels.data_ptr<int64_t>()),
           n_int4 * sizeof(int64_t));
    f.read(reinterpret_cast<char*>(int4_weights.data_ptr<int8_t>()),
           n_int4 * in_features * sizeof(int8_t));
    f.read(reinterpret_cast<char*>(int4_scales.data_ptr<at::Half>()),
           n_int4 * sizeof(at::Half));

    auto binary_channels = at::empty({n_binary}, int64_opts);
    f.read(reinterpret_cast<char*>(binary_channels.data_ptr<int64_t>()),
           n_binary * sizeof(int64_t));

    int64_t bits_bytes = (n_binary * in_features + 7) / 8;
    auto binary_bits = at::empty({bits_bytes}, uint8_opts);
    f.read(reinterpret_cast<char*>(binary_bits.data_ptr<uint8_t>()), bits_bytes);

    uint32_t n_blocks_u32, blocksize_u32;
    f.read(reinterpret_cast<char*>(&n_blocks_u32), sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(&blocksize_u32), sizeof(uint32_t));
    int64_t n_blocks = n_blocks_u32;
    int64_t blocksize = blocksize_u32;

    auto binary_scales = at::empty({n_binary, n_blocks}, half_opts);
    auto codebook_idx  = at::empty({n_binary, n_blocks}, uint8_opts);
    f.read(reinterpret_cast<char*>(binary_scales.data_ptr<at::Half>()),
           n_binary * n_blocks * sizeof(at::Half));
    f.read(reinterpret_cast<char*>(codebook_idx.data_ptr<uint8_t>()),
           n_binary * n_blocks * sizeof(uint8_t));

    // Read bias + check for NaN sentinel
    std::vector<uint16_t> bias_buf(out_features);
    f.read(reinterpret_cast<char*>(bias_buf.data()),
           out_features * sizeof(uint16_t));
    c10::optional<torch::Tensor> bias_opt;
    bool all_nan = true;
    for (int64_t i = 0; i < out_features; i++) {
        if (bias_buf[i] != kFp16NaN) { all_nan = false; break; }
    }
    if (!all_nan) {
        auto bias_t = at::empty({out_features}, half_opts);
        std::memcpy(bias_t.data_ptr<at::Half>(), bias_buf.data(),
                    out_features * sizeof(uint16_t));
        bias_opt = bias_t;
    }

    LoadedLayer out;
    out.layer_index   = hdr.layer_index;
    out.in_features   = in_features;
    out.out_features  = out_features;
    out.n_int4        = n_int4;
    out.n_binary      = n_binary;
    out.n_blocks      = n_blocks;
    out.blocksize     = blocksize;
    out.int4_channels = int4_channels;
    out.int4_weights  = int4_weights;
    out.int4_scales   = int4_scales;
    out.binary_channels = binary_channels;
    out.binary_bits   = binary_bits;
    out.binary_scales = binary_scales;
    out.codebook_idx  = codebook_idx;
    out.bias          = bias_opt;
    return out;
}

// ---------------------------------------------------------------------------
// Shared codebook read/write
// ---------------------------------------------------------------------------

void write_codebook_to_file(
    std::string path, torch::Tensor codebook   // [n_tiers, n_clusters, max_blocksize] fp16
) {
    TORCH_CHECK(codebook.scalar_type() == at::kHalf);
    TORCH_CHECK(codebook.dim() == 3);
    std::ofstream f(path, std::ios::binary);
    TORCH_CHECK(f.is_open(), "Cannot open file: ", path);

    CodebookHeader hdr{};
    hdr.magic        = kCodebookMagic;
    hdr.version      = kFormatVersion;
    hdr.n_tiers      = (uint32_t)codebook.size(0);
    hdr.n_clusters   = (uint32_t)codebook.size(1);
    hdr.max_blocksize = (uint32_t)codebook.size(2);
    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(codebook.data_ptr<at::Half>()),
            codebook.numel() * sizeof(at::Half));
}

torch::Tensor read_codebook_from_file(std::string path) {
    std::ifstream f(path, std::ios::binary);
    TORCH_CHECK(f.is_open(), "Cannot open file: ", path);
    CodebookHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    TORCH_CHECK(hdr.magic == kCodebookMagic, "Bad codebook magic");
    auto half_opts = at::TensorOptions().dtype(at::kHalf).device(at::kCPU);
    auto out = at::empty({(int64_t)hdr.n_tiers, (int64_t)hdr.n_clusters,
                          (int64_t)hdr.max_blocksize}, half_opts);
    f.read(reinterpret_cast<char*>(out.data_ptr<at::Half>()),
           out.numel() * sizeof(at::Half));
    return out;
}

}  // namespace fabq_rc
