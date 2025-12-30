import torch
import triton
import triton.language as tl


def get_configs():
    configs = []
    for num_stages in [2, 3, 4, 5]:
        for block_m in [16, 32, 64, 128]:
            for block_n in [32, 64, 128]:
                for block_k in [32, 64]:
                    for num_warps in [2, 4, 8]:
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_M": block_m,
                                    "BLOCK_N": block_n,
                                    "BLOCK_K": block_k,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                            )
                        )
    return configs


@triton.jit
def phi_rho_mm_kernel(
    # Pointers
    a_ptr,
    b_indices_ptr,
    codebook_ptr,
    c_ptr,
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes C = A @ Dequant(B_indices, Codebook)
    A: [M, K] (float16/bfloat16)
    B_indices: [K, N] (uint8)
    Codebook: [256] (float16/bfloat16)
    C: [M, N] (float16/bfloat16)
    """
    pid = tl.program_id(axis=0)

    # 1D launch grid
    # grid_m is not needed for coordinate calculation
    grid_n = tl.cdiv(N, BLOCK_N)

    # Standard tiling logic (pid -> pid_m, pid_n)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # Offsets for A and B
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_indices_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load codebook into SRAM (it's small, 256 elements)
    # We can rely on L1 cache for this since it's constant access pattern
    # But explicitly loading might be tricky if we don't know the dtype size per se in python
    # For now, we do indirect loads in the loop.

    for k in range(0, K, BLOCK_K):
        # Load A [BLOCK_M, BLOCK_K]
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)

        # Load B indices [BLOCK_K, BLOCK_N]
        b_idx = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0)

        # Dequantize B: lookup in codebook
        # b_idx is uint8 [0..255], codebook is [256]
        # values = codebook[indices]
        # Pointers to codebook: codebook_ptr + index
        b_val_ptrs = codebook_ptr + b_idx.to(tl.int32)
        b_val = tl.load(b_val_ptrs)

        # Dot product
        accumulator += tl.dot(a, b_val)

        # Advance
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)  # Casting back to half

    # Store C
    c_ptrs = c_ptr + (stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :])
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def run_phi_rho_mm(a, b_indices, codebook):
    # Checks
    assert a.is_cuda and b_indices.is_cuda and codebook.is_cuda
    assert a.ndim == 2 and b_indices.ndim == 2
    assert a.shape[1] == b_indices.shape[0], "K dimension mismatch"

    M, K = a.shape
    K, N = b_indices.shape

    # Alloc Output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Grid
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    # Configs
    # In a real scenario we'd use autotuner. For prototype we might pick a safe one or use the list
    # Let's try to just run one config for "correctness" first, then autotune.
    # We'll use a simplified autotuner call or manual launch for now to avoid dependency hell if autotuner imports fail

    # Using manual launch for the prototype "first pass"
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    num_warps = 4
    num_stages = 2

    phi_rho_mm_kernel[grid](
        a,
        b_indices,
        codebook,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_indices.stride(0),
        b_indices.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return c
