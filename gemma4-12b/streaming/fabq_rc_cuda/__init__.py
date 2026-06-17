"""fabq_rc_cuda - Python wrapper for the FABQ-RC CUDA extension.

This is the importable Python module that the streaming notebook uses. The
heavy lifting (CUDA kernel launches, file I/O) is in the C++/CUDA extension
imported as `fabq_rc_cuda._C`. This file provides:

  - QuantizedLinear: an nn.Module that wraps a FABQ-RC layer and runs the
    kernel forward pass. The module does NOT store an FP16 weight matrix.

  - FABQRCModel: a tiny nn.Module container that swaps FABQ-RC layers into a
    loaded HF model. Used by the streaming notebook to convert a standard
    Gemma 4 model on-the-fly.

  - load_layer_from_file / save_layer_to_file: thin wrappers around the C++
    I/O for the pre-quantized shards.

  - load_codebook / save_codebook: same, for the shared codebook.

Design notes:
  - The forward pass in QuantizedLinear calls the CUDA kernel directly. No
    intermediate FP16 weight materialization.
  - For non-CUDA tensors (e.g. during unit tests on CPU), we fall back to a
    pure-PyTorch implementation that reconstructs the FP16 weights for the
    matmul. This is a reference implementation, not a fast path. The CUDA
    kernel is the production path.
"""

from .quantized_linear import QuantizedLinear
from .model import FABQRCModel, quantize_model_in_place
from .io import (
    save_layer_to_file,
    load_layer_from_file,
    save_codebook,
    load_codebook,
)
from .kmeans import build_codebook
from .fisher import fisher_pass
from .quant_pipeline import select_blocksize_per_layer, allocate_precision

# The C++/CUDA extension. If it's not built yet, the wrappers above will
# raise an informative error at first use (not at import time), so the
# package can be imported in environments where the extension isn't built
# (e.g. CI lint, code review).
try:
    from . import _C
    CUDA_AVAILABLE = True
except ImportError:
    _C = None
    CUDA_AVAILABLE = False

__all__ = [
    "QuantizedLinear",
    "FABQRCModel",
    "quantize_model_in_place",
    "save_layer_to_file",
    "load_layer_from_file",
    "save_codebook",
    "load_codebook",
    "build_codebook",
    "fisher_pass",
    "select_blocksize_per_layer",
    "allocate_precision",
    "CUDA_AVAILABLE",
]

__version__ = "0.1.0"
