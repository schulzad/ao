from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass(frozen=True)
class PhiRhoCodebookConfig:
    """Configuration for deterministic phi/rho codebook quantization.

    Designed to be device-agnostic (CPU/MPS). The codebook is generated
    deterministically and assignments are computed via bucketize over midpoints.
    """

    num_codebook_entries: int = 256
    axis: Optional[int] = None  # If None, global codebook; axis-wise not yet implemented
    codebook_dtype: torch.dtype = torch.float32
    device: Optional[Union[str, torch.device]] = None

    def __post_init__(self):
        if self.num_codebook_entries < 2:
            raise ValueError("num_codebook_entries must be >= 2")

