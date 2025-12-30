import time
import torch
import math

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from torchao.quantization.prototype.phi_rho import PhiRhoCodebookConfig, PhiRhoCodebookQuantizedTensor

def bench(fn, iters=5):
    # Warmup
    fn()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    
    t0 = time.time()
    for _ in range(iters):
        fn()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    return (time.time() - t0) / iters

@torch.no_grad()
def int8_dynamic_quant(x: torch.Tensor):
    # Fair comparison: Includes finding min/max (scale/zp) calculation + quantization
    min_val, max_val = x.min(), x.max()
    scale = (max_val - min_val) / 255.0
    zero_point = (-min_val / scale).round().to(torch.int32)
    # Clamp zero_point to uint8 range
    zero_point = torch.clamp(zero_point, 0, 255)
    return torch.quantize_per_tensor(x, scale.item(), zero_point.item(), torch.quint8)

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Running on {device.upper()}")
    
    # Standard LLM-like matrix sizes
    shapes = [
        (4096, 4096),   # ~16M params
        (8192, 4096),   # ~33M params
        (8192, 8192)    # ~67M params
    ]
    
    results = []
    
    for M, N in shapes:
        print(f"Benchmarking Matrix {M}x{N}...")
        x = torch.randn(M, N, device=device)
        num_el = M * N
        
        # --- Phi-Rho ---
        cfg = PhiRhoCodebookConfig(num_codebook_entries=256) # 8-bit equivalent
        phi_s = bench(lambda: PhiRhoCodebookQuantizedTensor.from_float(x, cfg))
        
        # --- Int8 (MinMax + Quantize) ---
        # Note: torch.quantize_per_tensor on MPS might fallback or not be implemented efficiently, considering CPU mostly
        # If device is MPS, we might need to be careful. Phi-Rho supports MPS.
        # Let's ensure x is on the device.
        int8_s = bench(lambda: int8_dynamic_quant(x))
        
        # --- Float8 Cast (Reference) ---
        # Simple cast, usually memory bound
        f8_s = bench(lambda: x.to(torch.float8_e4m3fn))
        
        results.append({
            "Shape": f"{M}x{N}",
            "Params": f"{num_el/1e6:.1f}M",
            "Phi-Rho (s)": phi_s,
            "Int8 (s)": int8_s,
            "Float8 (s)": f8_s,
            "Speedup vs Int8": int8_s / phi_s,
            "Speedup vs Float8": f8_s / phi_s
        })

    if HAS_TABULATE:
        print(tabulate(results, headers="keys", floatfmt=".4f"))
    else:
        print(results)

if __name__ == '__main__':
    main()
