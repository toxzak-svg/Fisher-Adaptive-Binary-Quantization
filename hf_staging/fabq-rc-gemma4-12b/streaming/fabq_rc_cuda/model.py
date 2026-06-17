"""FABQRCModel - swap FABQ-RC layers into a standard HF model.

Used by the streaming notebook: load the BF16 model, quantize it layer by
layer as the shards stream in, swap the nn.Linear modules out for
QuantizedLinear modules. The swapped model runs natively on the CUDA
kernels.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .quantized_linear import QuantizedLinear


def _is_skippable(name: str) -> bool:
    """Layers we never quantize. Returns True for embeddings, lm_head, etc."""
    n = name.lower()
    if "embed" in n:
        return True
    if "lm_head" in n:
        return True
    if "norm" in n and "layernorm" not in n and "rmsnorm" not in n:
        # RMSNorm has no .weight we care about; embed/lm_head covered above
        return True
    return False


def quantize_model_in_place(
    model: nn.Module,
    allocation: Dict[str, Dict[int, str]],
    blocksize_results: Dict[str, int],
    codebook: torch.Tensor,
    fisher_scores: Optional[Dict[str, torch.Tensor]] = None,
    keep_bf16_in_vram: bool = True,
    progress: bool = True,
) -> nn.Module:
    """Quantize every nn.Linear in the model in place.

    For each layer:
      1. Read the BF16 weight from the existing module
      2. Call the C++ quantizer to produce int4 + binary buffers
      3. Replace the module with a QuantizedLinear holding the buffers
      4. Optionally free the BF16 from VRAM

    Args:
        model: the loaded HF model (modified in place)
        allocation: layer_name -> {channel_idx: 'int4' | 'binary'}
        blocksize_results: layer_name -> blocksize int
        codebook: [n_clusters, max_blocksize] fp16, the shared k-means codebook
        fisher_scores: unused in v1, kept for API compatibility
        keep_bf16_in_vram: if True, keep the original BF16 in the module
                          attribute .bf16_weight for debugging; if False, free it
        progress: whether to print progress

    Returns:
        The same model, with nn.Linear modules replaced by QuantizedLinear.
    """
    from . import _C
    from tqdm.auto import tqdm

    # Collect target layers
    targets: List[tuple] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _is_skippable(name):
            continue
        if name not in allocation:
            continue
        targets.append((name, module))

    if progress:
        print(f"Quantizing {len(targets)} Linear layers to FABQ-RC...")

    codebook_cpu = codebook.detach().cpu().to(torch.float32)

    for name, module in tqdm(targets, disable=not progress, desc="FABQ-RC quant"):
        weight = module.weight.data.float().cpu()
        alloc = allocation[name]
        bs = blocksize_results.get(name, 128)

        int4_chs = torch.tensor(
            sorted([ch for ch, p in alloc.items() if p == "int4"]),
            dtype=torch.long,
        )
        binary_chs = torch.tensor(
            sorted([ch for ch, p in alloc.items() if p == "binary"]),
            dtype=torch.long,
        )

        # Call C++ quantizer
        int4_w, int4_s, bin_bits, bin_s, cb_idx = _C.quantize_weight_matrix(
            weight.contiguous(), int4_chs, binary_chs, bs, codebook_cpu,
        )

        # Get bias
        bias = module.bias.detach().cpu().to(torch.float16) if module.bias is not None else None

        # Replace the module
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        new_mod = QuantizedLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            int4_channels=int4_chs,
            int4_weights=int4_w,
            int4_scales=int4_s,
            binary_channels=binary_chs,
            binary_bits=bin_bits,
            binary_scales=bin_s,
            codebook_idx=cb_idx,
            codebook=codebook.detach().cpu().to(torch.float16),
            blocksize=bs,
            bias=bias,
        )
        setattr(parent, child_name, new_mod)

        if not keep_bf16_in_vram:
            del module.weight
            if module.bias is not None:
                del module.bias

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model


class FABQRCModel(nn.Module):
    """Thin wrapper that holds a quantized model + the shared codebook.

    The streaming notebook wraps the HF model in this after quantization,
    so that the model's forward method can also return timing info.
    """

    def __init__(self, base_model: nn.Module, codebook: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.codebook = codebook

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def num_quantized_layers(self) -> int:
        from .quantized_linear import QuantizedLinear
        return sum(1 for m in self.base_model.modules()
                   if isinstance(m, QuantizedLinear))

    def compressed_size_gb(self) -> float:
        """Total bytes used by the compressed buffers (no FP16 weights)."""
        from .quantized_linear import QuantizedLinear
        total = 0
        for m in self.base_model.modules():
            if isinstance(m, QuantizedLinear):
                for buf_name in ("int4_weights", "int4_scales",
                                 "binary_bits", "binary_scales",
                                 "codebook_idx", "int4_channels",
                                 "binary_channels", "row_to_int4", "row_to_binary"):
                    buf = getattr(m, buf_name, None)
                    if buf is not None:
                        total += buf.numel() * buf.element_size()
                if m.bias is not None:
                    total += m.bias.numel() * m.bias.element_size()
        return total / 1e9
