"""quant_pipeline - the per-layer blocksize + precision allocation logic."""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


BS_CANDIDATES = [64, 128, 256, 512]
BS_PENALTIES = {64: 1.5, 128: 1.0, 256: 0.85, 512: 0.75}


def _is_target(name: str) -> bool:
    n = name.lower()
    if "embed" in n: return False
    if "lm_head" in n: return False
    if "gate" in n or "router" in n: return False
    return True


def _blocksize_recon_error(weights: np.ndarray, blocksize: int, fisher: np.ndarray) -> float:
    out_c, in_c = weights.shape
    total = 0.0
    for start in range(0, in_c, blocksize):
        end = min(start + blocksize, in_c)
        block = weights[:, start:end]
        scale = float(block.std()) + 1e-8
        block_q = np.where(block > 0, 1.0, -1.0) * scale
        recon = float(((block - block_q) ** 2).mean())
        block_fisher = float(fisher[start:end].mean())
        total += block_fisher * recon
    return total * BS_PENALTIES.get(blocksize, 1.0)


def _select_best_blocksize(weights: np.ndarray, fisher: np.ndarray) -> int:
    best_b, best_err = BS_CANDIDATES[0], float("inf")
    for b in BS_CANDIDATES:
        err = _blocksize_recon_error(weights, b, fisher)
        if err < best_err:
            best_err, best_b = err, b
    return best_b


def select_blocksize_per_layer(
    model: nn.Module,
    fisher_scores: dict,
) -> dict:
    """For each target layer, pick the blocksize that minimizes Fisher-weighted
    reconstruction error. Returns: dict[layer_name, blocksize]."""
    results = {}
    for name, module in tqdm(list(model.named_modules()), desc="Blocksize sweep"):
        if not _is_target(name):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if name not in fisher_scores:
            continue
        weights = module.weight.data.float().cpu().numpy()
        fisher = fisher_scores[name].float().cpu().numpy()
        results[name] = _select_best_blocksize(weights, fisher)
    return results


def allocate_precision(
    fisher_scores: dict,
    int4_fraction: float = 0.05,
) -> dict:
    """For each layer, sort channels by Fisher and assign int4 vs binary.

    Returns: dict[layer_name, {channel_idx: 'int4' | 'binary'}]
    """
    allocation = {}
    for name, fisher in fisher_scores.items():
        if not _is_target(name):
            continue
        if fisher.dim() == 0:
            fisher = fisher.unsqueeze(0)
        out_ch = fisher.shape[0]
        n_int4 = max(1, int(out_ch * int4_fraction))
        if out_ch <= 1:
            n_int4 = 1
        order = torch.argsort(fisher, descending=True)
        alloc = {int(ch): "int4" if rank < n_int4 else "binary"
                 for rank, ch in enumerate(order)}
        allocation[name] = alloc
    return allocation
