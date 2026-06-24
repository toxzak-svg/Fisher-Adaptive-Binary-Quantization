// bindings.cpp - pybind11 glue between Python and the C++/CUDA extension.

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fabq_rc_format.h"

// Forward declarations - the implementations live in fabq_rc_quant.cpp,
// fabq_rc_gemm.cu (v1), and fabq_rc_gemm_v2.cu (v2).
namespace fabq_rc {

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

LoadedLayer read_layer_from_file(std::string path);

std::vector<torch::Tensor> quantize_weight_matrix(
    torch::Tensor weight,
    torch::Tensor int4_channels,
    torch::Tensor binary_channels,
    int64_t blocksize,
    torch::Tensor codebook
);

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
);

void write_codebook_to_file(std::string path, torch::Tensor codebook);

torch::Tensor read_codebook_from_file(std::string path);

// v1 kernels (scalar, retained as reference / parity baseline)
torch::Tensor fabq_rc_gemm_int4(
    torch::Tensor x, torch::Tensor int4_w, torch::Tensor int4_scales,
    torch::Tensor row_to_int4, torch::Tensor y
);

torch::Tensor fabq_rc_gemm_binary(
    torch::Tensor x, torch::Tensor binary_bits,
    torch::Tensor binary_scales, torch::Tensor codebook_idx,
    torch::Tensor codebook, torch::Tensor row_to_binary,
    torch::Tensor y,
    int64_t n_blocks, int64_t blocksize, int64_t n_clusters, int64_t max_blocksize
);

torch::Tensor fabq_rc_gemm_mixed(
    torch::Tensor x,
    torch::Tensor int4_w, torch::Tensor int4_scales,
    torch::Tensor binary_bits, torch::Tensor binary_scales,
    torch::Tensor codebook_idx, torch::Tensor codebook,
    torch::Tensor row_to_int4, torch::Tensor row_to_binary,
    torch::Tensor y,
    int64_t n_blocks, int64_t blocksize, int64_t n_clusters, int64_t max_blocksize
);

void fabq_rc_add_bias(torch::Tensor y, torch::Tensor bias);

// v2 kernels (vectorized + tensor-core + embedding). v2_* mirror the v1
// signatures but add an optional bias and a v2_embed_lookup entry point.
torch::Tensor v2_gemm_int4(
    torch::Tensor x, torch::Tensor int4_w, torch::Tensor int4_scales,
    torch::Tensor row_to_int4,
    c10::optional<torch::Tensor> bias_opt,
    torch::Tensor y
);

torch::Tensor v2_gemm_binary(
    torch::Tensor x, torch::Tensor binary_bits,
    torch::Tensor binary_scales, torch::Tensor codebook_idx,
    torch::Tensor codebook, torch::Tensor row_to_binary,
    c10::optional<torch::Tensor> bias_opt,
    torch::Tensor y,
    int64_t n_blocks, int64_t blocksize, int64_t n_clusters, int64_t max_blocksize
);

torch::Tensor v2_gemm_mixed(
    torch::Tensor x,
    torch::Tensor int4_w, torch::Tensor int4_scales,
    torch::Tensor binary_bits, torch::Tensor binary_scales,
    torch::Tensor codebook_idx, torch::Tensor codebook,
    torch::Tensor row_to_int4, torch::Tensor row_to_binary,
    c10::optional<torch::Tensor> bias_opt,
    torch::Tensor y,
    int64_t n_blocks, int64_t blocksize, int64_t n_clusters, int64_t max_blocksize
);

torch::Tensor v2_embed_lookup(
    torch::Tensor token_ids,
    torch::Tensor int4_w, torch::Tensor int4_scales,
    torch::Tensor embed_int4_idx,
    torch::Tensor binary_bits, torch::Tensor binary_scales,
    torch::Tensor codebook_idx, torch::Tensor codebook,
    torch::Tensor embed_bin_idx,
    torch::Tensor y,
    int64_t n_blocks, int64_t blocksize, int64_t n_clusters, int64_t max_blocksize
);

// Wrapper around read_layer_from_file that returns a pybind11::dict (since
// LoadedLayer has c10::optional members that are awkward to return directly).
pybind11::dict read_layer_from_file_py(std::string path);

}  // namespace fabq_rc

namespace fabq_rc {

pybind11::dict read_layer_from_file_py(std::string path) {
    auto L = read_layer_from_file(path);
    pybind11::dict d;
    d["layer_index"]   = L.layer_index;
    d["in_features"]   = L.in_features;
    d["out_features"]  = L.out_features;
    d["n_int4"]        = L.n_int4;
    d["n_binary"]      = L.n_binary;
    d["n_blocks"]      = L.n_blocks;
    d["blocksize"]     = L.blocksize;
    d["int4_channels"] = L.int4_channels;
    d["int4_weights"]  = L.int4_weights;
    d["int4_scales"]   = L.int4_scales;
    d["binary_channels"] = L.binary_channels;
    d["binary_bits"]   = L.binary_bits;
    d["binary_scales"] = L.binary_scales;
    d["codebook_idx"]  = L.codebook_idx;
    if (L.bias.has_value() && L.bias->defined()) {
        d["bias"] = *L.bias;
    } else {
        d["bias"] = pybind11::none();
    }
    return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FABQ-RC: native-quantized inference CUDA extension. "
              "The forward pass operates on the compressed FABQ-RC format "
              "(int4 channels + binary bits + codebook indices) without ever "
              "materializing the full FP16 weight matrix.";

    // Quantization (CPU)
    m.def("quantize_weight_matrix", &fabq_rc::quantize_weight_matrix,
          "Quantize a FP32 weight matrix to FABQ-RC components. "
          "Returns [int4_w, int4_s, bin_bits, bin_s, cb_idx] as CPU tensors.",
          pybind11::arg("weight"),
          pybind11::arg("int4_channels"),
          pybind11::arg("binary_channels"),
          pybind11::arg("blocksize"),
          pybind11::arg("codebook"));

    m.def("write_layer_to_file", &fabq_rc::write_layer_to_file,
          "Write a single FABQ-RC quantized layer to a .bin file.",
          pybind11::arg("path"),
          pybind11::arg("layer_index"),
          pybind11::arg("in_features"),
          pybind11::arg("out_features"),
          pybind11::arg("int4_channels"),
          pybind11::arg("int4_weights"),
          pybind11::arg("int4_scales"),
          pybind11::arg("binary_channels"),
          pybind11::arg("binary_bits"),
          pybind11::arg("binary_scales"),
          pybind11::arg("codebook_idx"),
          pybind11::arg("blocksize"),
          pybind11::arg("bias") = pybind11::none());

    m.def("read_layer_from_file", &fabq_rc::read_layer_from_file_py,
          "Read a FABQ-RC quantized layer from a .bin file. "
          "Returns dict with all the layer's tensors and metadata.");

    m.def("write_codebook_to_file", &fabq_rc::write_codebook_to_file,
          "Write the shared k-means codebook to a .bin file.",
          pybind11::arg("path"), pybind11::arg("codebook"));

    m.def("read_codebook_from_file", &fabq_rc::read_codebook_from_file,
          "Read the shared k-means codebook from a .bin file.",
          pybind11::arg("path"));

    // Inference (CUDA)
    m.def("fabq_rc_gemm_int4", &fabq_rc::fabq_rc_gemm_int4,
          "Forward pass for a 100% int4 layer. Writes output to y in-place.",
          pybind11::arg("x"),
          pybind11::arg("int4_w"),
          pybind11::arg("int4_scales"),
          pybind11::arg("row_to_int4"),
          pybind11::arg("y"));

    m.def("fabq_rc_gemm_binary", &fabq_rc::fabq_rc_gemm_binary,
          "Forward pass for a 100% binary layer. Writes output to y in-place.",
          pybind11::arg("x"),
          pybind11::arg("binary_bits"),
          pybind11::arg("binary_scales"),
          pybind11::arg("codebook_idx"),
          pybind11::arg("codebook"),
          pybind11::arg("row_to_binary"),
          pybind11::arg("y"),
          pybind11::arg("n_blocks"),
          pybind11::arg("blocksize"),
          pybind11::arg("n_clusters"),
          pybind11::arg("max_blocksize"));

    m.def("fabq_rc_gemm_mixed", &fabq_rc::fabq_rc_gemm_mixed,
          "Forward pass for a mixed int4 + binary layer.",
          pybind11::arg("x"),
          pybind11::arg("int4_w"),
          pybind11::arg("int4_scales"),
          pybind11::arg("binary_bits"),
          pybind11::arg("binary_scales"),
          pybind11::arg("codebook_idx"),
          pybind11::arg("codebook"),
          pybind11::arg("row_to_int4"),
          pybind11::arg("row_to_binary"),
          pybind11::arg("y"),
          pybind11::arg("n_blocks"),
          pybind11::arg("blocksize"),
          pybind11::arg("n_clusters"),
          pybind11::arg("max_blocksize"));

    m.def("fabq_rc_add_bias", &fabq_rc::fabq_rc_add_bias,
          "Add a bias vector to the output tensor in-place.",
          pybind11::arg("y"), pybind11::arg("bias"));

    // ---- v2 kernels (vectorized + tensor-core + embedding) ----
    // v2 is a drop-in faster replacement for v1. Same numerical answer
    // within fp16 tolerance, same Python interface shape. The Python side
    // (QuantizedLinear) prefers v2 when available.
    m.def("v2_gemm_int4", &fabq_rc::v2_gemm_int4,
          "v2: vectorized / tensor-core int4 GEMM. Forward pass for a "
          "100% int4 layer. Writes output to y in-place. Bias is optional "
          "(pass None).",
          pybind11::arg("x"),
          pybind11::arg("int4_w"),
          pybind11::arg("int4_scales"),
          pybind11::arg("row_to_int4"),
          pybind11::arg("bias") = pybind11::none(),
          pybind11::arg("y"));

    m.def("v2_gemm_binary", &fabq_rc::v2_gemm_binary,
          "v2: vectorized binary-only GEMM with coalesced bit access. "
          "Forward pass for a 100% binary layer. Bias is optional.",
          pybind11::arg("x"),
          pybind11::arg("binary_bits"),
          pybind11::arg("binary_scales"),
          pybind11::arg("codebook_idx"),
          pybind11::arg("codebook"),
          pybind11::arg("row_to_binary"),
          pybind11::arg("bias") = pybind11::none(),
          pybind11::arg("y"),
          pybind11::arg("n_blocks"),
          pybind11::arg("blocksize"),
          pybind11::arg("n_clusters"),
          pybind11::arg("max_blocksize"));

    m.def("v2_gemm_mixed", &fabq_rc::v2_gemm_mixed,
          "v2: vectorized mixed int4 + binary GEMM. Forward pass for the "
          "general FABQ-RC layer (mixed allocation). Bias is optional.",
          pybind11::arg("x"),
          pybind11::arg("int4_w"),
          pybind11::arg("int4_scales"),
          pybind11::arg("binary_bits"),
          pybind11::arg("binary_scales"),
          pybind11::arg("codebook_idx"),
          pybind11::arg("codebook"),
          pybind11::arg("row_to_int4"),
          pybind11::arg("row_to_binary"),
          pybind11::arg("bias") = pybind11::none(),
          pybind11::arg("y"),
          pybind11::arg("n_blocks"),
          pybind11::arg("blocksize"),
          pybind11::arg("n_clusters"),
          pybind11::arg("max_blocksize"));

    m.def("v2_embed_lookup", &fabq_rc::v2_embed_lookup,
          "v2: quantized embedding lookup. Reads compressed embeddings "
          "(int4 channels + binary bits + codebook) directly without "
          "materializing an FP16 embedding table.",
          pybind11::arg("token_ids"),
          pybind11::arg("int4_w"),
          pybind11::arg("int4_scales"),
          pybind11::arg("embed_int4_idx"),
          pybind11::arg("binary_bits"),
          pybind11::arg("binary_scales"),
          pybind11::arg("codebook_idx"),
          pybind11::arg("codebook"),
          pybind11::arg("embed_bin_idx"),
          pybind11::arg("y"),
          pybind11::arg("n_blocks"),
          pybind11::arg("blocksize"),
          pybind11::arg("n_clusters"),
          pybind11::arg("max_blocksize"));

    // Version of the v2 path. Exposed so Python can decide whether to use
    // v2 or fall back to v1. v2 requires SM 8.0+ for the tensor-core path;
    // on older GPUs the scalar fallback inside v2 still runs.
    m.attr("V2_AVAILABLE") = true;

    // Format constants exposed for Python-side validation
    m.attr("LAYER_MAGIC")    = (uint64_t)fabq_rc::kLayerMagic;
    m.attr("CODEBOOK_MAGIC") = (uint64_t)fabq_rc::kCodebookMagic;
    m.attr("FORMAT_VERSION") = (uint64_t)fabq_rc::kFormatVersion;
    m.attr("CODEBOOK_TIERS")     = (int64_t)fabq_rc::kCodebookTiers;
    m.attr("CODEBOOK_CLUSTERS")  = (int64_t)fabq_rc::kCodebookClusters;
    m.attr("CODEBOOK_MAX_BLOCKS")= (int64_t)fabq_rc::kCodebookMaxBlocks;
}
}  // namespace fabq_rc
