"""build_codebook - k-means clustering of binary quantization residuals.

Pure-PyTorch / sklearn implementation. Called once during the
bucket-build step. The result is the shared codebook that all FABQ-RC
layers reference.
"""

from __future__ import annotations
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


def build_codebook(
    model,
    allocation: dict,
    blocksize_results: dict,
    n_clusters: int = 64,
    max_samples: int = 16384,
    max_blocksize: int = 512,
) -> torch.Tensor:
    """Build a shared k-means codebook from binary quantization residuals.

    Returns a tensor of shape [n_clusters, max_blocksize], dtype float16.
    """
    all_residuals = []
    sample_count = 0
    max_bs_seen = 0

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name not in allocation:
            continue
        weights = module.weight.detach().to(torch.float32).cpu().numpy()
        bs = blocksize_results.get(name, 128)
        max_bs_seen = max(max_bs_seen, bs)
        binary_chs = [ch for ch, p in allocation[name].items() if p == "binary"]
        if not binary_chs:
            continue

        step = max(1, len(binary_chs) // 20)
        for ch in binary_chs[::step]:
            for start in range(0, weights.shape[1], bs):
                end = min(start + bs, weights.shape[1])
                block = weights[ch, start:end]
                std = float(block.std()) + 1e-8
                block_q = np.where(block > 0, 1.0, -1.0) * std
                residual = (block - block_q).flatten()
                padded = np.pad(residual, (0, max_bs_seen - len(residual)))
                all_residuals.append(padded)
                sample_count += 1
                if sample_count >= max_samples:
                    break
            if sample_count >= max_samples:
                break
        if sample_count >= max_samples:
            break

    if not all_residuals:
        return torch.zeros(n_clusters, max_blocksize, dtype=torch.float16)

    # Pad all samples to the global max so the kmeans sees consistent dims
    global_max = max(r.shape[0] for r in all_residuals)
    all_residuals = [
        np.pad(r, (0, global_max - r.shape[0])) for r in all_residuals
    ]
    residuals_array = np.stack(all_residuals).astype(np.float32)

    # Drop NaN/Inf rows defensively
    mask = np.all(np.isfinite(residuals_array), axis=1)
    clean = residuals_array[mask]
    if clean.size == 0:
        return torch.zeros(n_clusters, max_blocksize, dtype=torch.float16)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42, batch_size=1024, n_init=3,
    )
    kmeans.fit(clean)

    # If the codebook's natural width is less than max_blocksize, pad with 0
    cb = kmeans.cluster_centers_.astype(np.float32)
    if cb.shape[1] < max_blocksize:
        cb = np.pad(cb, ((0, 0), (0, max_blocksize - cb.shape[1])))
    elif cb.shape[1] > max_blocksize:
        cb = cb[:, :max_blocksize]

    return torch.from_numpy(cb).to(torch.float16)
