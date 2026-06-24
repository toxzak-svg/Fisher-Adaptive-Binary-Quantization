"""Tests for the v2 CUDA kernels (fabq_rc_gemm_v2.cu).

v2 should produce numerically equivalent answers to v1 (within fp16 tolerance)
and to the PyTorch reference. Tests cover:

  - v2_int4 forward: 100% int4 layer (scalar path + tensor-core path)
  - v2_binary forward: 100% binary layer (vectorized bit access)
  - v2_mixed forward: mixed int4 + binary (fused bias)
  - v1 vs v2 parity: same input -> same output within fp16 tolerance
  - v2_embed_lookup: quantized embedding reconstruction

Tests are skipped if CUDA is not available. They also skip the tensor-core
kernel on architectures where it isn't compiled (older SM).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from fabq_rc_cuda.quantized_linear import QuantizedLinear
from fabq_rc_cuda.kmeans import build_codebook


def _make_codebook(n_clusters=64, max_blocksize=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_clusters, max_blocksize, generator=g, dtype=torch.float16) * 0.01


def _quantize_one_layer(weight, codebook, int4_frac=0.05, blocksize=128, seed=0):
    """Quantize a single FP32 weight matrix to FABQ-RC components."""
    from fabq_rc_cuda import _C
    g = torch.Generator().manual_seed(seed)
    out_f, in_f = weight.shape
    n_int4 = max(1, int(out_f * int4_frac))
    n_binary = out_f - n_int4
    perm = torch.randperm(out_f, generator=g)
    int4_chs = perm[:n_int4].long()
    binary_chs = perm[n_int4:].long()

    codebook_f32 = codebook.float()
    int4_w, int4_s, bin_bits, bin_s, cb_idx = _C.quantize_weight_matrix(
        weight.float().contiguous(),
        int4_chs, binary_chs, blocksize, codebook_f32,
    )
    return int4_chs, int4_w, int4_s, binary_chs, bin_bits, bin_s, cb_idx


def _make_layer(weight, codebook, int4_frac=0.1, blocksize=128, bias=None, seed=0):
    """Quantize + wrap as a CUDA QuantizedLinear."""
    int4_chs, int4_w, int4_s, binary_chs, bin_bits, bin_s, cb_idx = \
        _quantize_one_layer(weight, codebook, int4_frac, blocksize, seed)
    out_f, in_f = weight.shape
    layer = QuantizedLinear(
        in_features=in_f, out_features=out_f,
        int4_channels=int4_chs, int4_weights=int4_w, int4_scales=int4_s,
        binary_channels=binary_chs, binary_bits=bin_bits,
        binary_scales=bin_s, codebook_idx=cb_idx,
        codebook=codebook, blocksize=blocksize, bias=bias,
    )
    layer = layer.cuda()
    layer.int4_weights = layer.int4_weights.cuda()
    layer.int4_scales = layer.int4_scales.cuda()
    layer.binary_bits = layer.binary_bits.cuda()
    return layer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_v2_int4_only_layer():
    """100% int4 layer: v2 output == PyTorch reference within fp16."""
    torch.manual_seed(0)
    in_f, out_f = 256, 256
    weight = torch.randn(out_f, in_f, dtype=torch.float32)
    x = torch.randn(4, in_f, dtype=torch.float16, device="cuda")
    codebook = _make_codebook(max_blocksize=128).cuda()

    int4_chs = torch.arange(out_f, dtype=torch.long)
    binary_chs = torch.tensor([], dtype=torch.long)

    from fabq_rc_cuda import _C
    int4_w, int4_s, _, _, _ = _C.quantize_weight_matrix(
        weight.float().contiguous(),
        int4_chs, binary_chs, 128, codebook.float().cpu(),
    )
    int4_w = int4_w.cuda(); int4_s = int4_s.cuda()

    layer = QuantizedLinear(
        in_features=in_f, out_features=out_f,
        int4_channels=int4_chs, int4_weights=int4_w, int4_scales=int4_s,
        binary_channels=binary_chs,
        binary_bits=torch.zeros(0, dtype=torch.uint8),
        binary_scales=torch.zeros(0, 0, dtype=torch.float16),
        codebook_idx=torch.zeros(0, 0, dtype=torch.uint8),
        codebook=codebook, blocksize=128, bias=None,
    ).cuda()

    # Force v2
    layer._use_v2_kernel = True
    y_v2 = layer(x)
    # Reference
    layer._use_v2_kernel = False
    layer._use_cuda_kernel = False
    y_ref = layer(x)

    diff = (y_v2 - y_ref).abs().max().item()
    assert diff < 1e-3, f"v2 int4 mismatch: max diff {diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_v2_binary_only_layer():
    """100% binary layer: v2 output matches PyTorch reference."""
    torch.manual_seed(1)
    in_f, out_f = 256, 256
    weight = torch.randn(out_f, in_f, dtype=torch.float32)
    x = torch.randn(2, in_f, dtype=torch.float16, device="cuda")
    codebook = _make_codebook(max_blocksize=128)

    layer = _make_layer(weight, codebook, int4_frac=0.0, blocksize=128, seed=1)

    layer._use_v2_kernel = True
    y_v2 = layer(x)
    layer._use_v2_kernel = False
    layer._use_cuda_kernel = False
    y_ref = layer(x)

    diff = (y_v2 - y_ref).abs().max().item()
    assert diff < 0.5, f"v2 binary mismatch: max diff {diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_v2_mixed_layer_with_bias():
    """Mixed layer + bias: v2 fused bias path matches v1 bias path."""
    torch.manual_seed(2)
    in_f, out_f = 512, 512
    weight = torch.randn(out_f, in_f, dtype=torch.float32) * 0.1
    bias = torch.randn(out_f, dtype=torch.float32) * 0.05
    x = torch.randn(8, in_f, dtype=torch.float16, device="cuda")
    codebook = _make_codebook(max_blocksize=128)

    layer = _make_layer(weight, codebook, int4_frac=0.1, blocksize=128, bias=bias.half(), seed=2)

    layer._use_v2_kernel = True
    y_v2 = layer(x)
    layer._use_v2_kernel = False
    y_v1 = layer(x)

    diff = (y_v2 - y_v1).abs().max().item()
    assert diff < 1e-2, f"v2 vs v1 mixed+bias mismatch: max diff {diff}"

    # Reference
    layer._use_cuda_kernel = False
    y_ref = layer(x)
    diff = (y_v2 - y_ref).abs().max().item()
    assert diff < 0.5, f"v2 vs reference mismatch: max diff {diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_v2_int4_tensor_core_path():
    """Tensor-core path (B*T >= 16) produces same answer as scalar path."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    # Check compute capability >= 7.0 (WMMA requirement)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    if cc_major < 7:
        pytest.skip(f"needs SM 7.0+ (got {cc_major}.{cc_minor})")

    torch.manual_seed(3)
    in_f, out_f = 512, 512  # divisible by 16 and 128
    weight = torch.randn(out_f, in_f, dtype=torch.float32)
    x = torch.randn(32, in_f, dtype=torch.float16, device="cuda")  # B*T = 32 (TC path)
    codebook = _make_codebook(max_blocksize=128).cuda()

    int4_chs = torch.arange(out_f, dtype=torch.long)
    binary_chs = torch.tensor([], dtype=torch.long)

    from fabq_rc_cuda import _C
    int4_w, int4_s, _, _, _ = _C.quantize_weight_matrix(
        weight.float().contiguous(),
        int4_chs, binary_chs, 128, codebook.float().cpu(),
    )
    int4_w = int4_w.cuda(); int4_s = int4_s.cuda()

    layer = QuantizedLinear(
        in_features=in_f, out_features=out_f,
        int4_channels=int4_chs, int4_weights=int4_w, int4_scales=int4_s,
        binary_channels=binary_chs,
        binary_bits=torch.zeros(0, dtype=torch.uint8),
        binary_scales=torch.zeros(0, 0, dtype=torch.float16),
        codebook_idx=torch.zeros(0, 0, dtype=torch.uint8),
        codebook=codebook, blocksize=128, bias=None,
    ).cuda()

    # v2 with B*T=32 should hit the tensor-core dispatch
    layer._use_v2_kernel = True
    y_v2 = layer(x)

    # v1 scalar (always scalar)
    layer._use_v2_kernel = False
    y_v1 = layer(x)

    diff = (y_v2 - y_v1).abs().max().item()
    assert diff < 0.05, f"v2 tensor-core vs v1 mismatch: max diff {diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_v2_embed_lookup_int4():
    """Quantized embedding lookup (int4 row) matches BF16 reference."""
    from fabq_rc_cuda import _C

    torch.manual_seed(4)
    vocab_size = 1000
    embed_dim = 128
    embed_weight = torch.randn(vocab_size, embed_dim, dtype=torch.float32) * 0.1

    # 100% int4 embeddings
    embed_int4_idx = torch.arange(vocab_size, dtype=torch.long, device="cuda")
    embed_bin_idx = torch.full((vocab_size,), -1, dtype=torch.long, device="cuda")

    int4_w, int4_s, _, _, _ = _C.quantize_weight_matrix(
        embed_weight.contiguous(),
        embed_int4_idx.cpu(), embed_bin_idx.cpu(),
        128, torch.zeros(64, 128),  # dummy codebook (unused for int4 path)
    )
    int4_w = int4_w.cuda()
    int4_s = int4_s.cuda()

    token_ids = torch.tensor([0, 17, 42, 999], dtype=torch.long, device="cuda")
    y = torch.empty(token_ids.size(0), embed_dim, dtype=torch.float16, device="cuda")

    # v2 embed lookup (note: no binary inputs)
    _C.v2_embed_lookup(
        token_ids, int4_w, int4_s, embed_int4_idx,
        torch.zeros(0, dtype=torch.uint8, device="cuda"),
        torch.zeros(0, 0, dtype=torch.float16, device="cuda"),
        torch.zeros(0, 0, dtype=torch.uint8, device="cuda"),
        torch.zeros(64, 128, dtype=torch.float16, device="cuda"),
        embed_bin_idx, y,
        0, 128, 64, 128,
    )

    # Reference: just look up the quantized rows.
    expected = (int4_w.cpu().to(torch.float32) * int4_s.cpu().to(torch.float32).unsqueeze(-1))
    expected = expected[token_ids.cpu()].to(torch.float16).cuda()

    diff = (y - expected).abs().max().item()
    assert diff < 1e-2, f"v2 embed lookup int4 mismatch: max diff {diff}"


if __name__ == "__main__":
    # Run without pytest
    print("Running test_v2_int4_only_layer...")
    test_v2_int4_only_layer()
    print("  PASS")
    print("Running test_v2_binary_only_layer...")
    test_v2_binary_only_layer()
    print("  PASS")
    print("Running test_v2_mixed_layer_with_bias...")
    test_v2_mixed_layer_with_bias()
    print("  PASS")
    print("Running test_v2_int4_tensor_core_path...")
    test_v2_int4_tensor_core_path()
    print("  PASS")
    print("Running test_v2_embed_lookup_int4...")
    test_v2_embed_lookup_int4()
    print("  PASS")
    print("\nAll v2 tests passed.")
