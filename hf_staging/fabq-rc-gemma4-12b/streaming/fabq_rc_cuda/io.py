"""I/O wrappers for FABQ-RC layer files and the shared codebook."""

from __future__ import annotations
import torch
from typing import Optional


def save_layer_to_file(
    path: str,
    layer_index: int,
    in_features: int, out_features: int,
    int4_channels: torch.Tensor,
    int4_weights: torch.Tensor,
    int4_scales: torch.Tensor,
    binary_channels: torch.Tensor,
    binary_bits: torch.Tensor,
    binary_scales: torch.Tensor,
    codebook_idx: torch.Tensor,
    blocksize: int,
    bias: Optional[torch.Tensor] = None,
) -> None:
    """Write one FABQ-RC quantized layer to a .bin file."""
    from . import _C
    _C.write_layer_to_file(
        path, layer_index, in_features, out_features,
        int4_channels.contiguous(),
        int4_weights.contiguous(),
        int4_scales.contiguous(),
        binary_channels.contiguous(),
        binary_bits.contiguous(),
        binary_scales.contiguous(),
        codebook_idx.contiguous(),
        blocksize,
        bias.contiguous() if bias is not None else None,
    )


def load_layer_from_file(path: str) -> dict:
    """Read one FABQ-RC quantized layer from a .bin file.

    Returns a dict with all the layer's tensors and metadata.
    """
    from . import _C
    return _C.read_layer_from_file(path)


def save_codebook(path: str, codebook: torch.Tensor) -> None:
    """Write the shared k-means codebook to a .bin file."""
    from . import _C
    _C.write_codebook_to_file(path, codebook.contiguous())


def load_codebook(path: str) -> torch.Tensor:
    """Read the shared k-means codebook from a .bin file."""
    from . import _C
    return _C.read_codebook_from_file(path)
