## Phi/Rho Deterministic Codebook Quantization (Prototype)

Deterministic codebook quantization replacing k-means in `CodebookQuantizedTensor.from_float()`.

- Deterministic Chebyshev-like codebook over data range
- O(N) bucketize assignment via midpoints (no O(N*K) distance matrix)
- CPU/MPS friendly; no CUDA/Triton required; `torch.compile`-friendly

### Motivation
K-means is costly (O(N·K) passes), has unpredictable convergence, and can be memory-heavy. This replaces it with a linear-time, deterministic alternative with competitive or better reconstruction error for common NN data distributions.

### Algorithm
- Codebook: `K` cosine-spaced nodes mapped to `[min(x), max(x)]`, sorted ascending.
- Assignment: compute midpoints and use `torch.bucketize(values, edges)` to choose nearest code.
- Dequant: `codebook[indices]`.

### Complexity
- Time: O(N) + O(K) setup.
- Memory: O(K) extra.
- Deterministic: no randomness; repeatable results.

### Benchmarks (M2, N=1e6, K=256)
- `phi_rho_from_float_s`: ~0.044 s
- `phi_rho_mse`: ~2.8e-4
- local k-means baseline: ~0.56 s, MSE ~5.6e-4

### Usage
```python
from torchao.quantization.prototype.phi_rho import (
    PhiRhoCodebookConfig, PhiRhoCodebookQuantizedTensor
)
import torch

x = torch.randn(1_000_000)
q = PhiRhoCodebookQuantizedTensor.from_float(x, PhiRhoCodebookConfig(num_codebook_entries=256))
y = q.to_float()
```

One-liner on a module:
```python
from torchao.quantization import quantize_
from torchao.quantization.prototype.phi_rho import PhiRhoCodebookConfig
import torch.nn as nn

m = nn.Linear(4096, 4096, bias=False)
quantize_(m, PhiRhoCodebookConfig(num_codebook_entries=256))
```

### Limitations / Future Work
- Per-axis codebooks not yet implemented.
- Optional single-step Lloyd refinement could be added (still cheap).
- Later: packing/ops fusion for end-to-end inference speedups.


