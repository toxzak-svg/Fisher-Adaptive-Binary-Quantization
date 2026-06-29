import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fabq_rc_cuda.kmeans import build_codebook


def test_build_codebook_skips_incomplete_tail_blocks():
    model = torch.nn.Sequential(torch.nn.Linear(5, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, -1.0, 1.0, -1.0, 100.0]]))

    allocation = {"0": {0: "binary"}}
    blocksize_results = {"0": 4}

    codebook = build_codebook(
        model,
        allocation,
        blocksize_results,
        n_clusters=1,
        max_samples=16,
        max_blocksize=4,
    )

    assert torch.allclose(codebook[0], torch.zeros(4, dtype=torch.float16), atol=1e-3)
