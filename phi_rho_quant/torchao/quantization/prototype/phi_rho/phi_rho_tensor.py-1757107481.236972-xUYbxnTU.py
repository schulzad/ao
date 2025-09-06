from dataclasses import dataclass
from typing import Optional

import torch

from .phi_rho_config import PhiRhoCodebookConfig
from .phi_rho_kernels import generate_phi_rho_codebook, quantize_values, lloyd_refine_1d


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
    def from_float(cls, values: torch.Tensor, config: PhiRhoCodebookConfig) -> "PhiRhoCodebookQuantizedTensor":
        device = torch.device(config.device) if config.device is not None else values.device
        if config.axis is None:
            flat = values.detach().to(device=device).view(-1)
            min_val = flat.min()
            max_val = flat.max()
            if torch.isclose(min_val, max_val):
                codebook = torch.stack([min_val, max_val]).to(device=device, dtype=config.codebook_dtype)
                indices = torch.zeros_like(flat, dtype=torch.long)
            else:
                codebook = generate_phi_rho_codebook(min_val, max_val, config.num_codebook_entries, dtype=config.codebook_dtype, device=device)
                if config.lloyd_refine_steps > 0:
                    codebook = lloyd_refine_1d(flat, codebook, steps=config.lloyd_refine_steps)
                indices, _ = quantize_values(flat, codebook)
            return cls(
                indices=indices.view(values.shape),
                codebook=codebook,
                original_shape=values.shape,
                original_dtype=values.dtype,
            )
        # Per-axis codebook
        axis = config.axis
        x = values.detach().to(device=device)
        D = x.dim()
        permute_order = [axis] + [d for d in range(D) if d != axis]
        # Move axis to dim 0 and flatten the rest
        x_perm = x.permute(*permute_order)
        num_slices = x_perm.size(0)
        x_flat = x_perm.reshape(num_slices, -1)
        K = config.num_codebook_entries
        codebooks = []  # list of per-slice [K]
        all_indices = torch.empty_like(x_flat, dtype=torch.long)
        for i in range(num_slices):
            v = x_flat[i]
            mn, mx = v.min(), v.max()
            if torch.isclose(mn, mx):
                cb = torch.full((K,), mn.to(dtype=config.codebook_dtype), device=device, dtype=config.codebook_dtype)
                idx = torch.zeros_like(v, dtype=torch.long)
            else:
                cb = generate_phi_rho_codebook(mn, mx, K, dtype=config.codebook_dtype, device=device)
                if config.lloyd_refine_steps > 0:
                    cb = lloyd_refine_1d(v, cb, steps=config.lloyd_refine_steps)
                idx, _ = quantize_values(v, cb)
            # Offset indices into concatenated codebook
            all_indices[i] = idx + i * K
            codebooks.append(cb)
        # Concatenate per-slice codebooks to 1D
        codebook = torch.cat(codebooks, dim=0)
        # Reshape indices back to permuted shape, then invert permutation
        indices_perm = all_indices.view_as(x_perm)
        # Build inverse permutation Q such that Q[permute_order[i]] = i
        inv_perm = [0] * D
        for i, d in enumerate(permute_order):
            inv_perm[d] = i
        indices = indices_perm.permute(*inv_perm)
        return cls(
            indices=indices,
            codebook=codebook,
            original_shape=values.shape,
            original_dtype=values.dtype,
        )

    @torch.no_grad()
    def to_float(self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor:
        out = self.codebook[self.indices]
        if dtype is None:
            dtype = self.original_dtype
        if device is None:
            device = self.codebook.device
        return out.to(dtype=dtype, device=device)


