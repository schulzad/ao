"""PhiRhoLinear: A drop-in replacement for nn.Linear using Phi-Rho quantization."""

import torch
import torch.nn as nn

from .phi_rho_config import PhiRhoCodebookConfig
from .phi_rho_tensor import PhiRhoCodebookQuantizedTensor

# Try to import Triton kernel, fall back to dequant if not available
try:
    from .triton import run_phi_rho_mm
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


class PhiRhoLinear(nn.Module):
    """
    A quantized Linear layer using Phi-Rho codebook quantization.
    
    Stores weights as uint8 indices + a 256-entry codebook.
    Uses a fused Triton kernel for inference when available.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Placeholder for quantized weights (will be set by from_float)
        self.register_buffer("weight_indices", None)
        self.register_buffer("codebook", None)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
    
    @classmethod
    def from_float(cls, linear: nn.Linear, config: PhiRhoCodebookConfig = None):
        """Create a PhiRhoLinear from an existing nn.Linear."""
        if config is None:
            config = PhiRhoCodebookConfig(num_codebook_entries=256)
        
        device = linear.weight.device
        dtype = linear.weight.dtype
        
        # Create the quantized layer
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=device,
            dtype=dtype,
        )
        
        # Quantize the weight
        q = PhiRhoCodebookQuantizedTensor.from_float(linear.weight, config)
        
        # Store indices. Use uint8 if valid, else int16 (or int32)
        if config.num_codebook_entries <= 256:
            layer.weight_indices = q.indices.to(torch.uint8)
        elif config.num_codebook_entries <= 32768:
            # PyTorch doesn't fully support uint16, so we use int16
            layer.weight_indices = q.indices.to(torch.int16)
        else:
            layer.weight_indices = q.indices.to(torch.int32)

        layer.codebook = q.codebook.to(dtype)
        
        # Copy bias if present
        if linear.bias is not None:
            layer.bias = nn.Parameter(linear.bias.clone())
        
        return layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x @ W.T + b
        
        Uses fused Triton kernel if available, otherwise dequantizes.
        """
        # x: [*, in_features]
        # weight_indices: [out_features, in_features]
        # codebook: [256]
        
        batch_shape = x.shape[:-1]
        x_2d = x.view(-1, self.in_features)  # [B, in_features]
        
        if HAS_TRITON and x.is_cuda:
            # Transpose weight indices for matmul: [in_features, out_features]
            w_indices_t = self.weight_indices.t().contiguous()
            out = run_phi_rho_mm(x_2d, w_indices_t, self.codebook)
        else:
            # Fallback: dequantize and use standard matmul
            weight = self.codebook[self.weight_indices.long()]
            out = torch.nn.functional.linear(x_2d, weight, None)
        
        out = out.view(*batch_shape, self.out_features)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def extra_repr(self) -> str:
        s = f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
        if self.weight_indices is not None:
            s += f", index_dtype={self.weight_indices.dtype}"
        return s
