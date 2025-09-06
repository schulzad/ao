import torch

from torchao.quantization.prototype.phi_rho import (
    PhiRhoCodebookConfig, PhiRhoCodebookQuantizedTensor
)


def test_roundtrip_shape_and_dtype_cpu():
    x = torch.randn(1024, dtype=torch.float32, device='cpu')
    cfg = PhiRhoCodebookConfig(num_codebook_entries=64)
    q = PhiRhoCodebookQuantizedTensor.from_float(x, cfg)
    y = q.to_float()
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_constant_tensor():
    x = torch.ones(256, dtype=torch.float32)
    cfg = PhiRhoCodebookConfig(num_codebook_entries=64)
    q = PhiRhoCodebookQuantizedTensor.from_float(x, cfg)
    y = q.to_float()
    assert torch.allclose(y, x)


def test_mps_if_available():
    if not torch.backends.mps.is_available():
        return
    x = torch.randn(2048, dtype=torch.float32, device='mps')
    cfg = PhiRhoCodebookConfig(num_codebook_entries=128)
    q = PhiRhoCodebookQuantizedTensor.from_float(x, cfg)
    y = q.to_float(device=torch.device('mps'))
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_lloyd_refinement_improves_mse_cpu():
    x = torch.randn(50000, dtype=torch.float32, device='cpu')
    cfg0 = PhiRhoCodebookConfig(num_codebook_entries=128, lloyd_refine_steps=0)
    cfg1 = PhiRhoCodebookConfig(num_codebook_entries=128, lloyd_refine_steps=1)
    q0 = PhiRhoCodebookQuantizedTensor.from_float(x, cfg0)
    q1 = PhiRhoCodebookQuantizedTensor.from_float(x, cfg1)
    mse0 = torch.mean((q0.to_float() - x) ** 2)
    mse1 = torch.mean((q1.to_float() - x) ** 2)
    assert mse1 <= mse0 * 1.01  # allow tiny noise margin


def test_per_axis_codebook_shape_cpu():
    x = torch.randn(16, 128, dtype=torch.float32)
    cfg = PhiRhoCodebookConfig(num_codebook_entries=64, axis=0)
    q = PhiRhoCodebookQuantizedTensor.from_float(x, cfg)
    y = q.to_float()
    assert y.shape == x.shape



