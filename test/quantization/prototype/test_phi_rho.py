import torch

from torchao.quantization.prototype.phi_rho import (
    PhiRhoCodebookConfig,
    PhiRhoCodebookQuantizedTensor,
)


def test_roundtrip_shape_and_dtype_cpu():
    x = torch.randn(1024, dtype=torch.float32, device="cpu")
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
    x = torch.randn(2048, dtype=torch.float32, device="mps")
    cfg = PhiRhoCodebookConfig(num_codebook_entries=128)
    q = PhiRhoCodebookQuantizedTensor.from_float(x, cfg)
    y = q.to_float(device=torch.device("mps"))
    assert y.shape == x.shape
    assert y.dtype == x.dtype
