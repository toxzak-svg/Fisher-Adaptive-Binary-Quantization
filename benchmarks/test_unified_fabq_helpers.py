import subprocess
import sys
from pathlib import Path

from benchmark_unified_fabq import (
    estimate_mix_bpw,
    precision_mix_for_target,
)


BENCHMARK_DIR = Path(__file__).resolve().parent


def run_torch_check(source: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=str(BENCHMARK_DIR),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_precision_mix_for_target_tracks_requested_budget():
    mix = precision_mix_for_target(3.0)

    assert set(mix) == {"int8", "int4", "int2", "binary"}
    assert abs(sum(mix.values()) - 1.0) < 1e-12
    assert estimate_mix_bpw(mix) <= 3.05
    assert mix["int4"] > mix["binary"]


def test_weighted_mse_uses_imatrix_input_importance():
    run_torch_check(
        """
import torch
from benchmark_unified_fabq import weighted_mse
weight = torch.zeros(2, 3)
recon = torch.tensor([[1.0, 1.0, 1.0], [0.0, 2.0, 0.0]])
importance = torch.tensor([10.0, 1.0, 1.0])
assert weighted_mse(weight, recon, importance) > weighted_mse(weight, recon, torch.ones(3))
"""
    )


def test_allocate_precision_by_damage_promotes_high_damage_rows():
    run_torch_check(
        """
import torch
from benchmark_unified_fabq import allocate_precision_by_damage, precision_mix_for_target
weight = torch.tensor([
    [0.01, -0.01, 0.01, -0.01],
    [4.0, -3.0, 2.0, -1.0],
    [0.02, 0.02, -0.02, -0.02],
    [0.03, -0.03, 0.03, -0.03],
])
allocation = allocate_precision_by_damage(weight, torch.ones(4), precision_mix_for_target(3.0))
assert allocation[1] in {"int8", "int4"}
assert allocation.count("binary") >= 1
"""
    )


def test_quantize_symmetric_two_bit_is_more_accurate_than_binary():
    run_torch_check(
        """
import torch
from benchmark_unified_fabq import quantize_symmetric, weighted_mse
weight = torch.tensor([[0.1, -0.4, 0.9, -1.7]])
importance = torch.ones(4)
binary = quantize_symmetric(weight, bits=1, input_importance=importance, blocksize=4)
int2 = quantize_symmetric(weight, bits=2, input_importance=importance, blocksize=4)
assert weighted_mse(weight, int2, importance) < weighted_mse(weight, binary, importance)
"""
    )


def test_block_residual_correction_reduces_binary_error():
    run_torch_check(
        """
import torch
from benchmark_unified_fabq import apply_block_residual_correction, quantize_symmetric, weighted_mse
weight = torch.tensor([[0.2, 0.4, 1.2, 1.4]])
importance = torch.ones(4)
recon = quantize_symmetric(weight, bits=1, input_importance=importance, blocksize=4)
corrected = apply_block_residual_correction(weight, recon, blocksize=4)
assert weighted_mse(weight, corrected, importance) < weighted_mse(weight, recon, importance)
"""
    )
