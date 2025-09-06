from typing import Any

import torch

from .phi_rho_config import PhiRhoCodebookConfig
from .phi_rho_tensor import PhiRhoCodebookQuantizedTensor


def apply_phi_rho_to_module(module: torch.nn.Module, config: PhiRhoCodebookConfig) -> torch.nn.Module:
    """Minimal adapter that replaces float weights with phi/rho codebook tensors.

    This is a stub; in TorchAO this would be registered with quantize_ flows.
    """
    for name, param in list(module.named_parameters(recurse=True)):
        if not param.requires_grad:
            continue
        q = PhiRhoCodebookQuantizedTensor.from_float(param.data, config)
        # Store as a simple attribute; a full integration would subclass tensor.
        setattr(module, f"{name}_phi_rho", q)
    return module


