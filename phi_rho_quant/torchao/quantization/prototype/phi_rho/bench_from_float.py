import time
import torch

from .phi_rho_config import PhiRhoCodebookConfig
from .phi_rho_tensor import PhiRhoCodebookQuantizedTensor


def bench(fn, iters=5):
    # warmup
    fn()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    return (time.time() - t0) / iters


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    x = torch.randn(2_000_000, device=device)
    cfg = PhiRhoCodebookConfig(num_codebook_entries=256)

    def run_phi_rho():
        PhiRhoCodebookQuantizedTensor.from_float(x, cfg)

    s = bench(run_phi_rho, iters=3)
    print({"phi_rho_from_float_s": s})


if __name__ == "__main__":
    main()


