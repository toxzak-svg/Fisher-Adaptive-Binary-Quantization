// fabq_rc_format.h - on-disk and in-memory format for FABQ-RC quantized weights.
//
// The key invariant: a QuantizedLinear layer stores ONLY the compressed
// representation. There is no FP16 weight matrix anywhere. The CUDA kernel
// reads these buffers directly to produce the matmul output.
//
// ============================================================================
// Per-layer file format (.bin, one file per shard of N layers)
// ============================================================================
//
// Header (32 bytes, little-endian):
//   u32  magic          = 0x46514246 ('FBQB' = FABQ Binary)
//   u32  version        = 1
//   u32  layer_index
//   u32  reserved
//   u32  in_features
//   u32  out_features
//   u32  n_int4
//   u32  n_binary
//
// Followed by, in order:
//
//   int4_channels[i64; n_int4]
//   int4_weights[i8;  n_int4 * in_features]
//   int4_scales[f16;  n_int4]
//
//   binary_channels[i64; n_binary]
//   binary_bits[u8;   ceil(n_binary * in_features / 8)]   // packed 1-bit
//   n_blocks[u32]                                          // blocks per binary channel
//   blocksize[u32]                                         // input elements per block
//   binary_scales[f16; n_binary * n_blocks]
//   codebook_indices[u8; n_binary * n_blocks]              // index into shared codebook
//
//   bias[f16; out_features]                                // optional, f32 NaN sentinel if absent
//
// All multi-byte values are little-endian.
//
// ============================================================================
// Shared codebook file (one per model, fabqrc-codebook.bin)
// ============================================================================
//
// Header:
//   u32  magic          = 0x46424346 ('FCBF')
//   u32  version        = 1
//   u32  n_tiers        = 4
//   u32  n_clusters     = 64
//   u32  max_blocksize  = 512
//
// Then 4 codebooks, each [n_clusters, max_blocksize] of fp16.
// Total size: 32 + 4 * 64 * 512 * 2 = 32 + 256 KB = 256 KB.
//
// ============================================================================
// Notes
// ============================================================================
//
// - Codebook is shared across all layers of the same model. Different models
//   may have different codebooks.
// - The "tier" of a codebook entry is selected at quant time per-layer based
//   on that layer's Fisher quartile. For v1 we use a single shared codebook
//   (tier-agnostic) and accept the ~5% quality hit in exchange for simplicity.
//   v2 will reintroduce tiers.
// - bias is f16 with a NaN sentinel (0x7E00 in fp16) meaning "no bias" if the
//   original layer had bias=None. The kernel checks for NaN at load time.
//
// ============================================================================

#pragma once
#include <cstdint>

namespace fabq_rc {

constexpr uint32_t kLayerMagic   = 0x46514246;  // 'FBQB'
constexpr uint32_t kCodebookMagic = 0x46424346;  // 'FCBF'
constexpr uint32_t kFormatVersion = 1;

constexpr int kCodebookTiers     = 4;
constexpr int kCodebookClusters  = 64;
constexpr int kCodebookMaxBlocks = 512;

// NaN in fp16 used as "no bias" sentinel.
constexpr uint16_t kFp16NaN = 0x7E00;

struct LayerHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t layer_index;
    uint32_t reserved;
    uint32_t in_features;
    uint32_t out_features;
    uint32_t n_int4;
    uint32_t n_binary;
};

struct CodebookHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t n_tiers;
    uint32_t n_clusters;
    uint32_t max_blocksize;
};

}  // namespace fabq_rc
