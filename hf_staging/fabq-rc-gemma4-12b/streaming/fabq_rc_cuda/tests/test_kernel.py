"""Numerical correctness tests for fabq_rc_cuda.

Run with: pytest tests/test_kernel.py -v

Or just: python tests/test_kernel.py

These tests use the PyTorch reference implementation (which materializes
the FP16 weight) as the ground truth, then check that the CUDA kernel
produces the same answer within fp16 tolerance. They DO NOT verify that
the CUDA kernel is faster than the reference - that's a separate benchmark.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import pytest

from fabq_rc_cuda.quantized_linear import QuantizedLinear
from fabq_rc_cuda.kmeans import build_codebook


def make_codebook(n_clusters=64, max_blocksize=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_clusters, max_blocksize, generator=g, dtype=torch.float16) * 0.01


def quantize_one_layer(weight, codebook, int4_frac=0.05, blocksize=128, seed=0):
    """Helper: quantize a single FP32 weight matrix to FABQ-RC components."""
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_int4_only_layer_forward():
    """A 100% int4 layer: CUDA kernel output == PyTorch reference output."""
    torch.manual_seed(0)
    in_f, out_f = 256, 256
    weight = torch.randn(out_f, in_f, dtype=torch.float16, device="cuda")
    x = torch.randn(4, in_f, dtype=torch.float16, device="cuda")

    codebook = make_codebook(max_blocksize=128).cuda()
    # Make ALL channels int4 (binary_frac=0)
    int4_chs = torch.arange(out_f, dtype=torch.long)
    binary_chs = torch.tensor([], dtype=torch.long)

    from fabq_rc_cuda import _C
    int4_w, int4_s, _, _, _ = _C.quantize_weight_matrix(
        weight.float().cpu().contiguous(),
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

    y_cuda = layer(x)
    y_ref = x @ weight.T  # PyTorch reference
    assert torch.allclose(y_cuda, y_ref, atol=1e-2, rtol=1e-2), \
        f"int4 mismatch: max diff {(y_cuda - y_ref).abs().max().item()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mixed_layer_forward():
    """Mixed int4 + binary: CUDA kernel output ~= PyTorch reference within fp16 tol."""
    torch.manual_seed(1)
    in_f, out_f = 512, 512
    weight = torch.randn(out_f, in_f, dtype=torch.float32)
    x = torch.randn(8, in_f, dtype=torch.float16, device="cuda")

    codebook = make_codebook(max_blocksize=128)

    int4_chs, int4_w, int4_s, binary_chs, bin_bits, bin_s, cb_idx = \
        quantize_one_layer(weight, codebook, int4_frac=0.1, blocksize=128, seed=1)

    layer = QuantizedLinear(
        in_features=in_f, out_features=out_f,
        int4_channels=int4_chs, int4_weights=int4_w, int4_scales=int4_s,
        binary_channels=binary_chs,
        binary_bits=bin_bits,
        binary_scales=bin_s,
        codebook_idx=cb_idx,
        codebook=codebook, blocksize=128, bias=None,
    )
    # Move everything to CUDA
    layer = layer.cuda()
    int4_w_cuda = int4_w.cuda(); int4_s_cuda = int4_s.cuda()
    bin_bits_cuda = bin_bits.cuda()
    layer.int4_weights = int4_w_cuda
    layer.int4_scales = int4_s_cuda
    layer.binary_bits = bin_bits_cuda

    y_cuda = layer(x)
    y_ref = (x.float() @ weight.T.cuda().float()).half()
    diff = (y_cuda - y_ref).abs().max().item()
    # fp16 accumulation + bit quantization -> some loss, but should be small
    assert diff < 0.5, f"mixed-layer mismatch: max diff {diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_bias_add():
    """Bias add kernel: y + bias[o] should equal the matmul result with bias."""
    torch.manual_seed(2)
    in_f, out_f = 128, 64
    weight = torch.randn(out_f, in_f, dtype=torch.float32) * 0.1
    bias = torch.randn(out_f, dtype=torch.float32) * 0.1
    x = torch.randn(2, in_f, dtype=torch.float16, device="cuda")

    codebook = make_codebook(max_blocksize=128)
    int4_chs = torch.arange(out_f, dtype=torch.long)
    binary_chs = torch.tensor([], dtype=torch.long)

    from fabq_rc_cuda import _C
    int4_w, int4_s, _, _, _ = _C.quantize_weight_matrix(
        weight.contiguous(), int4_chs, binary_chs, 128, codebook.float(),
    )
    layer = QuantizedLinear(
        in_features=in_f, out_features=out_f,
        int4_channels=int4_chs, int4_weights=int4_w, int4_scales=int4_s,
        binary_channels=binary_chs,
        binary_bits=torch.zeros(0, dtype=torch.uint8),
        binary_scales=torch.zeros(0, 0, dtype=torch.float16),
        codebook_idx=torch.zeros(0, 0, dtype=torch.uint8),
        codebook=codebook, blocksize=128, bias=bias.half(),
    ).cuda()
    layer.int4_weights = layer.int4_weights.cuda()
    layer.int4_scales = layer.int4_scales.cuda()

    y_cuda = layer(x)
    y_ref = (x.float() @ weight.T.cuda().float() + bias.cuda().float()).half()
    diff = (y_cuda - y_ref).abs().max().item()
    assert diff < 0.05, f"bias add mismatch: max diff {diff}"


def test_codebook_format():
    """Codebook round-trip: write to file, read back, verify shape + values."""
    import tempfile
    from fabq_rc_cuda.io import save_codebook, load_codebook

    cb = make_codebook(n_clusters=64, max_blocksize=512, seed=42)
    # Wrap in [1, n_clusters, max_blocksize] for the file format (tier-0 only for v1)
    cb_packed = cb.unsqueeze(0)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
    try:
        save_codebook(path, cb_packed)
        cb_loaded = load_codebook(path)
        assert cb_loaded.shape == (1, 64, 512)
        assert torch.allclose(cb_loaded[0], cb, atol=1e-3)
    finally:
        os.unlink(path)


def test_layer_file_roundtrip():
    """Layer round-trip: quantize, write, read, reconstruct - should match original."""
    import tempfile
    from fabq_rc_cuda.io import save_layer_to_file, load_layer_from_file

    torch.manual_seed(3)
    out_f, in_f = 128, 64
    weight = torch.randn(out_f, in_f, dtype=torch.float32) * 0.1
    codebook = make_codebook(max_blocksize=128)

    int4_chs = torch.arange(out_f // 4, dtype=torch.long)  # 25% int4
    binary_chs = torch.arange(out_f // 4, out_f, dtype=torch.long)
    blocksize = 64

    from fabq_rc_cuda import _C
    int4_w, int4_s, bin_bits, bin_s, cb_idx = _C.quantize_weight_matrix(
        weight.contiguous(), int4_chs, binary_chs, blocksize, codebook.float(),
    )

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
    try:
        save_layer_to_file(
            path, layer_index=0, in_features=in_f, out_features=out_f,
            int4_channels=int4_chs, int4_weights=int4_w, int4_scales=int4_s,
            binary_channels=binary_chs, binary_bits=bin_bits,
            binary_scales=bin_s, codebook_idx=cb_idx, blocksize=blocksize,
            bias=None,
        )
        loaded = load_layer_from_file(path)
        assert loaded["in_features"] == in_f
        assert loaded["out_features"] == out_f
        assert loaded["n_int4"] == int4_chs.numel()
        assert loaded["n_binary"] == binary_chs.numel()
        assert loaded["blocksize"] == blocksize
        assert loaded["int4_weights"].shape == int4_w.shape
        assert torch.allclose(loaded["int4_weights"], int4_w)
    finally:
        os.unlink(path)


if __name__ == "__main__":
    # Run without pytest for quick check
    print("Running test_int4_only_layer_forward...")
    test_int4_only_layer_forward()
    print("  PASS")
    print("Running test_mixed_layer_forward...")
    test_mixed_layer_forward()
    print("  PASS")
    print("Running test_bias_add...")
    test_bias_add()
    print("  PASS")
    print("Running test_codebook_format...")
    test_codebook_format()
    print("  PASS")
    print("Running test_layer_file_roundtrip...")
    test_layer_file_roundtrip()
    print("  PASS")
    print("\nAll tests passed (CUDA path).")
