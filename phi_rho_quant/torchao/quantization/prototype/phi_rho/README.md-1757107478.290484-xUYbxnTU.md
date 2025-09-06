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

### Hybrid refinement (optional)
Set `lloyd_refine_steps=1` on `PhiRhoCodebookConfig` to run a single 1D Lloyd step after deterministic init:
- Still O(N) per step; uses bucketize-based assignments.
- Typically reduces MSE further with small extra time.

### Per-axis codebooks (prototype)
Set `axis` to an integer dimension to quantize per-axis (e.g., `axis=0` for per-out-channel `Linear` weights). This prototype stores concatenated codebooks; a production version would pack per-axis codebooks efficiently.

### Benchmarks
Run realistic shapes and vectorized multi-layer suite:

- CPU with baseline (local k-means if TorchAO codebook unavailable):
```
RUN_DEVICE=cpu INCLUDE_BASELINE=1 python -m torchao.quantization.prototype.phi_rho.bench_realistic | cat
```
- MPS without baseline (avoids Metal command-buffer issues in baseline path):
```
RUN_DEVICE=mps INCLUDE_BASELINE=0 python -m torchao.quantization.prototype.phi_rho.bench_realistic | cat
```

### Results (CPU)
- linear_w_4k_4k, K=256: phi 0.0386s (MSE 3.43e-4), phi+Lloyd 0.091s (2.80e-4), baseline 0.346s (5.31e-4)
- linear_w_8k_4k, K=256: phi 0.0627s (3.51e-4), phi+Lloyd 0.165s (2.77e-4), baseline 0.725s (4.81e-4)
- act 8x4096, K=256: phi 0.00027s (1.86e-4), phi+Lloyd 0.00054s (1.40e-4), baseline 0.00156s (3.03e-4)
- act 32x4096, K=256: phi 0.00050s (2.12e-4), phi+Lloyd 0.00129s (1.73e-4), baseline 0.00466s (4.95e-4)
- 12-layer suite (K=256): phi total 0.570s, baseline total 4.302s

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
quantize_(m, PhiRhoCodebookConfig(num_codebook_entries=256, lloyd_refine_steps=1))
```

### Limitations / Future Work
- Per-axis codebooks not yet implemented.
- Optional single-step Lloyd refinement could be added (still cheap).
- Later: packing/ops fusion for end-to-end inference speedups.


