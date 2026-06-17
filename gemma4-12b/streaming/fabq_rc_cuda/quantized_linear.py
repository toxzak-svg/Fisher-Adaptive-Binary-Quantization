"""QuantizedLinear - the FABQ-RC nn.Module.

This is THE module that runs in the model. It stores only the compressed
buffers (int4 weights, binary bits, codebook indices, scales) - never an
FP16 weight matrix. The forward pass calls the CUDA kernel.

For CPU fallback / testing, the same module can also reconstruct an FP16
weight on the fly (one layer at a time, then discard). This is the
reference implementation; the CUDA kernel is what you want in production.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class QuantizedLinear(nn.Module):
    """An nn.Linear replacement that stores FABQ-RC compressed weights.

    The class is "duck-typed" to look like an nn.Linear from the outside:
    - .in_features, .out_features
    - .weight (raises if accessed - we never materialize it)
    - .bias (None or a 1-D tensor)
    - forward(x) returns the same shape as nn.Linear would
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        int4_channels: torch.Tensor,         # [n_int4]   int64
        int4_weights: torch.Tensor,          # [n_int4, in_features] int8
        int4_scales: torch.Tensor,           # [n_int4]   fp16
        binary_channels: torch.Tensor,       # [n_binary] int64
        binary_bits: torch.Tensor,           # packed uint8
        binary_scales: torch.Tensor,         # [n_binary, n_blocks] fp16
        codebook_idx: torch.Tensor,          # [n_binary, n_blocks] uint8
        codebook: torch.Tensor,              # [n_clusters, max_blocksize] fp16
        blocksize: int,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.blocksize = blocksize
        self.n_clusters = codebook.size(0)
        self.max_blocksize = codebook.size(1)

        # Channel maps: row index -> which int4/binary slot, or -1
        n_int4 = int4_channels.size(0)
        n_binary = binary_channels.size(0)
        row_to_int4 = torch.full((out_features,), -1, dtype=torch.long)
        row_to_binary = torch.full((out_features,), -1, dtype=torch.long)
        if n_int4 > 0:
            row_to_int4[int4_channels.long()] = torch.arange(n_int4, dtype=torch.long)
        if n_binary > 0:
            row_to_binary[binary_channels.long()] = torch.arange(n_binary, dtype=torch.long)

        # Register buffers so they move with .to(device)
        self.register_buffer("int4_weights", int4_weights)
        self.register_buffer("int4_scales", int4_scales)
        self.register_buffer("binary_bits", binary_bits)
        self.register_buffer("binary_scales", binary_scales)
        self.register_buffer("codebook_idx", codebook_idx)
        self.register_buffer("codebook", codebook)
        self.register_buffer("row_to_int4", row_to_int4)
        self.register_buffer("row_to_binary", row_to_binary)
        self.register_buffer("int4_channels", int4_channels)
        self.register_buffer("binary_channels", binary_channels)

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

        # Whether to use the CUDA kernel (True) or the PyTorch reference (False)
        self._use_cuda_kernel: bool = True

    @property
    def weight(self) -> torch.Tensor:
        """Pretend to have a weight. Always raises - we don't store one."""
        raise AttributeError(
            "QuantizedLinear does not store a weight matrix. "
            "The forward pass operates on the compressed representation directly. "
            "If you need an FP16 reconstruction (e.g. for testing or export), "
            "call .reconstruct_weight() - this allocates an FP16 tensor for the "
            "duration of one call, do not store the result."
        )

    def num_blocks(self) -> int:
        if self.binary_scales.numel() == 0:
            return 0
        return int(self.binary_scales.size(1))

    def reconstruct_weight(self) -> torch.Tensor:
        """Allocate an FP16 weight matrix from the compressed buffers.

        This is for testing, debugging, and GGUF export - NOT for the
        forward pass. Calling this defeats the point of FABQ-RC at runtime.
        """
        device = self.int4_weights.device
        dtype = torch.float16
        w = torch.zeros(self.out_features, self.in_features, dtype=dtype, device=device)

        # Int4 channels
        if self.int4_channels.numel() > 0:
            ch = self.int4_channels.to(device)
            iw = self.int4_weights.to(device).to(dtype)
            s = self.int4_scales.to(device).to(dtype)
            w[ch] = iw * s.unsqueeze(-1)

        # Binary channels: bit-unpack + per-block scale + codebook correction
        if self.binary_channels.numel() > 0:
            bch = self.binary_channels.to(device)
            n_binary = bch.size(0)
            in_features = self.in_features
            n_blocks = self.num_blocks()
            bs = self.blocksize
            cb = self.codebook.to(device).to(dtype)  # [n_clusters, max_blocksize]

            for k in range(n_binary):
                row = bch[k].item()
                bin_scales_k = self.binary_scales[k].to(device).to(dtype)  # [n_blocks]
                cb_idx_k = self.codebook_idx[k].to(device).long()         # [n_blocks]

                for blk in range(n_blocks):
                    blk_start = blk * bs
                    blk_end = min(blk_start + bs, in_features)
                    blk_len = blk_end - blk_start
                    scale = bin_scales_k[blk]
                    cb_id = cb_idx_k[blk].item()
                    cb_vec = cb[cb_id, :blk_len]

                    for i in range(blk_len):
                        bit_idx = k * in_features + blk_start + i
                        byte_idx = bit_idx >> 3
                        bit_off = bit_idx & 7
                        sign = 1 if (self.binary_bits[byte_idx].item() >> bit_off) & 1 else -1
                        w[row, blk_start + i] = sign * scale + cb_vec[i]
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - calls the CUDA kernel. NEVER materializes FP16 weight."""
        if self._use_cuda_kernel and x.is_cuda:
            return self._forward_cuda(x)
        else:
            return self._forward_reference(x)

    def _forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        from . import _C
        # Flatten leading dims: [..., in_features] -> [B*T, in_features]
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).to(torch.float16).contiguous()

        y = torch.empty(x_2d.size(0), self.out_features,
                        dtype=torch.float16, device=x.device)

        n_int4 = self.int4_channels.numel()
        n_binary = self.binary_channels.numel()

        if n_int4 > 0 and n_binary == 0:
            _C.fabq_rc_gemm_int4(
                x_2d, self.int4_weights, self.int4_scales,
                self.row_to_int4, y,
            )
        elif n_int4 == 0 and n_binary > 0:
            _C.fabq_rc_gemm_binary(
                x_2d, self.binary_bits, self.binary_scales, self.codebook_idx,
                self.codebook, self.row_to_binary, y,
                self.num_blocks(), self.blocksize, self.n_clusters, self.max_blocksize,
            )
        else:
            _C.fabq_rc_gemm_mixed(
                x_2d, self.int4_weights, self.int4_scales,
                self.binary_bits, self.binary_scales, self.codebook_idx,
                self.codebook, self.row_to_int4, self.row_to_binary, y,
                self.num_blocks(), self.blocksize, self.n_clusters, self.max_blocksize,
            )

        if self.bias is not None:
            _C.fabq_rc_add_bias(y, self.bias)

        return y.view(*orig_shape[:-1], self.out_features)

    def _forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Pure-PyTorch reference. Materializes the FP16 weight for the matmul.
        Used for CPU testing, debugging, and the times when the CUDA kernel
        isn't built yet. NOT the production path."""
        w = self.reconstruct_weight().to(x.dtype)
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        n_int4 = int(self.int4_channels.numel())
        n_binary = int(self.binary_channels.numel())
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"int4_channels={n_int4}, binary_channels={n_binary}, "
                f"blocksize={self.blocksize}, n_clusters={self.n_clusters}")
