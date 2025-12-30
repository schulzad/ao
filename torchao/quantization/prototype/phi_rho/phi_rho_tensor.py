from dataclasses import dataclass
from typing import Optional

import torch

from .phi_rho_config import PhiRhoCodebookConfig
from .phi_rho_kernels import generate_phi_rho_codebook, quantize_values


@dataclass
class PhiRhoCodebookQuantizedTensor:
    """A simple holder for codebook quantization results.

    This is not a Tensor subclass; it stores indices and codebook and can
    reconstruct the float tensor via to_float().
    """

    indices: torch.Tensor  # Long tensor with same shape as original values
    codebook: torch.Tensor  # Shape [K]
    original_shape: torch.Size
    original_dtype: torch.dtype

    @classmethod
    @torch.no_grad()
    def from_float(
        cls, values: torch.Tensor, config: PhiRhoCodebookConfig
    ) -> "PhiRhoCodebookQuantizedTensor":
        if config.axis is not None:
            raise NotImplementedError(
                "axis-wise codebook not yet implemented in prototype"
            )
        device = (
            torch.device(config.device) if config.device is not None else values.device
        )
        flat = values.detach().to(device=device)
        min_val = flat.min()
        max_val = flat.max()
        if torch.isclose(min_val, max_val):
            codebook = torch.stack([min_val, max_val]).to(
                device=device, dtype=config.codebook_dtype
            )
            indices = torch.zeros_like(flat, dtype=torch.long)
        else:
            codebook = generate_phi_rho_codebook(
                min_val,
                max_val,
                config.num_codebook_entries,
                dtype=config.codebook_dtype,
                device=device,
            )
            indices, _ = quantize_values(flat, codebook)
        return cls(
            indices=indices.view(values.shape),
            codebook=codebook,
            original_shape=values.shape,
            original_dtype=values.dtype,
        )

    @torch.no_grad()
    def to_float(
        self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        out = self.codebook[self.indices]
        if dtype is None:
            dtype = self.original_dtype
        if device is None:
            device = self.codebook.device
        return out.to(dtype=dtype, device=device)
