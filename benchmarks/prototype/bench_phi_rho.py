import time
import importlib
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
def local_kmeans_1d(values: torch.Tensor, k: int, iters: int = 8):
    # Simple linear init
    min_val = values.min()
    max_val = values.max()
    centers = torch.linspace(min_val, max_val, steps=k, device=values.device, dtype=values.dtype)
    for _ in range(iters):
        # Assign via bucketize midpoints for 1D
        edges = (centers[1:] + centers[:-1]) * 0.5
        assign = torch.bucketize(values, edges)
        
        # Update means
        sums = torch.zeros(k, device=values.device, dtype=values.dtype)
        counts = torch.zeros(k, device=values.device, dtype=torch.long)
        sums += torch.bincount(assign, weights=values, minlength=k)
        counts += torch.bincount(assign, minlength=k)
        
        # Avoid div by zero
        counts = torch.clamp_min(counts.to(values.dtype), 1.0)
        centers = sums / counts
        
        # Fill empty clusters with percentiles (simple heuristic)
        empty = counts <= 1.0  # treated as empty or single point
        if empty.any():
            # spread out empty centroids
            n_empty = empty.sum().item()
            q = torch.linspace(0, 1, n_empty + 2, device=values.device, dtype=values.dtype)[1:-1]
            centers[empty] = torch.quantile(values, q)
            centers, _ = torch.sort(centers)
            
    return centers

def get_data(dist_name, N, device):
    if dist_name == 'normal':
        return torch.randn(N, device=device)
    elif dist_name == 'uniform':
        return torch.rand(N, device=device) * 10 - 5
    elif dist_name == 'lognormal':
        return torch.exp(torch.randn(N, device=device))
    elif dist_name == 'bimodal':
        # Mixture of two Gaussians
        mask = torch.rand(N, device=device) > 0.5
        d1 = torch.randn(N, device=device) - 3
        d2 = torch.randn(N, device=device) + 3
        return torch.where(mask, d1, d2)
    else:
        raise ValueError(f"Unknown distribution {dist_name}")

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Running on {device.upper()}")
    
    N = 1_000_000
    K = 256
    distributions = ['normal', 'uniform', 'lognormal', 'bimodal']
    
    results = []
    
    for dist in distributions:
        print(f"Benchmarking {dist}...")
        x = get_data(dist, N, device)
        x_min, x_max = x.min().item(), x.max().item()
        
        # --- Phi-Rho ---
        cfg = PhiRhoCodebookConfig(num_codebook_entries=K)
        
        def run_phi_rho():
            PhiRhoCodebookQuantizedTensor.from_float(x, cfg)
            
        phi_s = bench(run_phi_rho)
        
        # Calc MSE
        q_phi = PhiRhoCodebookQuantizedTensor.from_float(x, cfg)
        phi_mse = torch.mean((q_phi.to_float() - x) ** 2).item()
        
        # --- Baseline (K-Means) ---
        # We'll use the local implementation for consistency unless import logic is restored
        def run_baseline():
            centers = local_kmeans_1d(x, K, iters=8)
            assign = torch.bucketize(x, (centers[1:] + centers[:-1]) * 0.5)
            # simulate construction cost
            _ = centers[assign]

        base_s = bench(run_baseline)
        
        # Calc MSE
        centers = local_kmeans_1d(x, K, iters=8)
        assign = torch.bucketize(x, (centers[1:] + centers[:-1]) * 0.5)
        base_mse = torch.mean((centers[assign] - x) ** 2).item()
        
        results.append({
            "Distribution": dist,
            "Range": f"[{x_min:.1f}, {x_max:.1f}]",
            "Phi-Rho (s)": phi_s,
            "Baseline (s)": base_s,
            "Speedup": base_s / phi_s,
            "Phi-Rho MSE": phi_mse,
            "Base MSE": base_mse,
            "MSE Ratio": phi_mse / base_mse
        })

    if HAS_TABULATE:
        print(tabulate(results, headers="keys", floatfmt=".5f"))
    else:
        # Simple markdown table fallback
        print("| Dist | Phi-Rho (s) | Base (s) | Speedup | Phi-Rho MSE | Base MSE | Ratio |")
        print("|---|---|---|---|---|---|---|")
        for r in results:
            print(f"| {r['Distribution']} | {r['Phi-Rho (s)']:.5f} | {r['Baseline (s)']:.5f} | {r['Speedup']:.2f}x | {r['Phi-Rho MSE']:.5f} | {r['Base MSE']:.5f} | {r['MSE Ratio']:.2f} |")

if __name__ == '__main__':
    main()
