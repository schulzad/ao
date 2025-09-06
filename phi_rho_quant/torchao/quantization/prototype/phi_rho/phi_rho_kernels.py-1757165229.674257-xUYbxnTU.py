from typing import Tuple

import torch


@torch.no_grad()
def generate_phi_rho_codebook(min_value: torch.Tensor, max_value: torch.Tensor, num_entries: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Generate a deterministic codebook in [min_value, max_value] using cosine-spaced nodes.

    We use Chebyshev-like nodes mapped to the data range for good coverage;
    this is deterministic and fast to compute on CPU/MPS.
    """
    if num_entries == 1:
        return min_value.to(device=device, dtype=dtype).view(1)
    # Chebyshev nodes in [-1, 1]
    k = torch.arange(1, num_entries + 1, device=device, dtype=dtype)
    nodes = torch.cos((2 * k - 1) * torch.pi / (2 * num_entries))
    # Map to [min, max]
    center = (min_value + max_value) * 0.5
    half_span = (max_value - min_value) * 0.5
    codebook = center + half_span * nodes
    # Ensure ascending order for bucketize boundaries
    codebook, _ = torch.sort(codebook)
    return codebook.to(device=device, dtype=dtype)


@torch.no_grad()
def compute_bin_edges_from_codebook(codebook: torch.Tensor) -> torch.Tensor:
    """Compute midpoints between adjacent codebook entries as bin edges for bucketize.

    Returns edges of length (K-1), ascending.
    """
    return (codebook[1:] + codebook[:-1]) * 0.5


@torch.no_grad()
def assign_via_bucketize(values: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Assign values to nearest codebook entries using bucketize over midpoints.

    This avoids O(N*K) distance matrices and scales linearly with N.
    """
    edges = compute_bin_edges_from_codebook(codebook)
    indices = torch.bucketize(values, edges)
    return indices


@torch.no_grad()
def quantize_values(values: torch.Tensor, codebook: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (indices, dequant) where dequant = codebook[indices]."""
    indices = assign_via_bucketize(values, codebook)
    dequant = codebook[indices]
    return indices, dequant


@torch.no_grad()
def lloyd_refine_1d(values: torch.Tensor, codebook: torch.Tensor, steps: int) -> torch.Tensor:
    """Run a few Lloyd iterations in 1D using bucketize assignments.

    This preserves O(N) per step and avoids full distance matrices.
    """
    if steps <= 0:
        return codebook
    centers = codebook
    for _ in range(steps):
        edges = compute_bin_edges_from_codebook(centers)
        assign = torch.bucketize(values, edges)
        sums = torch.bincount(assign, weights=values, minlength=centers.numel())
        counts = torch.bincount(assign, minlength=centers.numel())
        new_centers = sums / torch.clamp_min(counts.to(values.dtype), 1)
        empty = counts == 0
        if empty.any():
            mn, mx = values.min(), values.max()
            fill = torch.linspace(mn, mx, steps=empty.sum().item() + 2, device=values.device, dtype=values.dtype)[1:-1]
            new_centers[empty] = fill
        centers = torch.sort(new_centers).values
    return centers


