
import torch
import time
import triton

# Import the kernel wrapper
# Note: In a real run we need to make sure imports work. We assume PYTHONPATH is set.
from torchao.quantization.prototype.phi_rho.triton import run_phi_rho_mm

def bench(fn, iters=10):
    # Warmup
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters

def main():
    if not torch.cuda.is_available():
        print("Skipping benchmark: CUDA not available")
        return

    device = "cuda"
    dtype = torch.float16
    
    # Dimensions
    M = 4096
    N = 4096
    K = 4096
    
    print(f"Benchmarking Phi-Rho Inference MM ({M}x{K} @ {K}x{N}) on {torch.cuda.get_device_name(0)}")
    
    # Data
    a = torch.randn(M, K, device=device, dtype=dtype)
    # B Indices (Uint8) simulating quantized weights
    b_indices = torch.randint(0, 256, (K, N), device=device, dtype=torch.uint8)
    # Codebook (256, )
    codebook = torch.randn(256, device=device, dtype=dtype)
    
    # Reference (De-quantized)
    b_ref = codebook[b_indices.long()] # Materialize float matrix [K, N]
    
    # 1. Run Standard Matmul (Ideal Baseline - if weights were already float)
    # This measures pure compute bound.
    triton_time = bench(lambda: torch.matmul(a, b_ref))
    print(f"FP16 MM (Baseline): {triton_time*1000:.3f} ms")
    
    # 2. Run Phi-Rho Fused Kernel
    # This fuses the lookup. It should ideally be competitive with FP16 MM 
    # but wins on memory bandwidth (fetching uint8 instead of fp16).
    try:
        phi_rho_time = bench(lambda: run_phi_rho_mm(a, b_indices, codebook))
        print(f"Phi-Rho Fused MM:   {phi_rho_time*1000:.3f} ms")
        print(f"Speedup vs FP16:    {triton_time / phi_rho_time:.2f}x")
    except Exception as e:
        print(f"Phi-Rho Failed: {e}")
        
    # Check Correctness
    out_phi = run_phi_rho_mm(a, b_indices, codebook)
    out_ref = torch.matmul(a, b_ref)
    
    # Tolerances for half precision
    # Note: atomic Adds in triton vs PyTorch accum order might cause slight diffs
    if torch.allclose(out_phi, out_ref, atol=1e-2, rtol=1e-2):
        print("Correctness: PASS")
    else:
        diff = (out_phi - out_ref).abs().max().item()
        print(f"Correctness: FAIL (Max Diff: {diff})")

if __name__ == "__main__":
    main()
