"""Quantize only expert-valid rows, with the same FP8/BF16 rounding contract."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quant_valid(
    X,
    Q,
    S,
    L,
    M: tl.constexpr,
    N: tl.constexpr,
    BN: tl.constexpr,
    W: tl.constexpr,
    MASKED: tl.constexpr,
    ACT: tl.constexpr,
):
    e = tl.program_id(0)
    worker = tl.program_id(1)
    if MASKED:
        start = e.to(tl.int64) * M
        count = tl.load(L + e)
    else:
        start = tl.load(L + e).to(tl.int64)
        count = tl.load(L + e + 1) - start
    groups = tl.arange(0, BN // 128)
    cols = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    for row in range(worker, count, W):
        index = start + row
        if ACT:
            g = tl.load(X + index * (2 * N) + cols, cols < N, 0).to(tl.float32)
            u = tl.load(X + index * (2 * N) + N + cols, cols < N, 0).to(tl.float32)
            g = tl.minimum(g, 10.0)
            u = tl.maximum(-10.0, tl.minimum(u, 10.0))
            silu = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
            val = (silu * u).to(tl.bfloat16).to(tl.float32)
        else:
            val = tl.load(X + index * N + cols, cols < N, 0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(val), 1), 1.0e-10)
        scale = tl.div_rn(amax, 448.0)
        quant = tl.maximum(-448.0, tl.minimum(448.0, tl.div_rn(val, scale[:, None])))
        tl.store(Q + index * N + cols, quant, cols < N)
        tl.store(S + index * (N // 128) + groups, scale, groups < N // 128)


def quant_valid(x, layout, kind, G, activation=False, workers=4):
    """Quantize BF16 rows selected by device counts or contiguous offsets.

    For grouped_masked, layout contains G counts in [0, x.shape[0] / G].
    For grouped_contiguous, it contains G+1 monotonic offsets into x.
    Padding is left uninitialized and must not be consumed by later kernels.
    Activation fuses SwiGLU with clamp=10 and BF16 intermediate rounding.
    """
    if x.ndim != 2 or not x.is_contiguous() or x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError("Expected contiguous CUDA BF16 matrix")
    if G <= 0 or workers <= 0 or kind not in ("grouped_masked", "grouped_contiguous"):
        raise ValueError("Invalid expert geometry or layout kind")
    expected = G if kind == "grouped_masked" else G + 1
    if (
        layout.numel() != expected
        or layout.device != x.device
        or not layout.is_contiguous()
        or layout.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Expected contiguous device integer counts/offsets")
    if kind == "grouped_masked" and x.shape[0] % G:
        raise ValueError("Masked storage must contain equal capacity per expert")
    N = x.shape[-1] // (2 if activation else 1)
    if N <= 0 or N % 128 or x.shape[1] != N * (2 if activation else 1):
        raise ValueError("Quantization requires a width divisible by 128")
    q = torch.empty(x.shape[0], N, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(x.shape[0], N // 128, device=x.device, dtype=torch.float32)
    _quant_valid[(G, workers)](
        x,
        q,
        s,
        layout,
        x.shape[0] // G,
        N,
        triton.next_power_of_2(N),
        workers,
        kind == "grouped_masked",
        activation,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return q, s
