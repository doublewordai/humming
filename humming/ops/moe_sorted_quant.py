"""Fused SwiGLU/FP8 quantization over current GPU-resident expert indices."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quant(X, Q, S, IDS, P, M: tl.constexpr, N: tl.constexpr, CB: tl.constexpr, CTAS: tl.constexpr):
    pid = tl.program_id(0)
    chunks = tl.cdiv(N, CB)
    padded = tl.load(P)
    groups = tl.arange(0, CB // 128)
    lane = tl.arange(0, 128)
    for item in range(pid, padded * chunks, CTAS):
        rid = item // chunks
        chunk = item % chunks
        row = tl.load(IDS + rid).to(tl.int64)
        if (row >= 0) & (row < M):
            cols = chunk * CB + groups[:, None] * 128 + lane[None, :]
            g = tl.load(X + row * (2 * N) + cols, cols < N, 0).to(tl.float32)
            u = tl.load(X + row * (2 * N) + N + cols, cols < N, 0).to(tl.float32)
            g = tl.minimum(g, 10.0)
            u = tl.maximum(-10.0, tl.minimum(u, 10.0))
            silu = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
            val = (silu * u).to(tl.bfloat16).to(tl.float32)
            amax = tl.maximum(tl.max(tl.abs(val), 1), 1.0e-10)
            scale = tl.div_rn(amax, 448.0)
            quant = tl.maximum(-448.0, tl.minimum(448.0, tl.div_rn(val, scale[:, None])))
            tl.store(Q + row * N + cols, quant, cols < N)
            sg = chunk * (CB // 128) + groups
            tl.store(S + row * (N // 128) + sg, scale, sg < N // 128)


def quant_sorted(x, sorted_ids, padded, chunk=512):
    """Quantize unique live indexed rows; skip negative and end sentinels.

    padded is a device scalar in [0, sorted_ids.numel()]. Each valid row must
    occur once. Padding output is uninitialized. Uses clamped SwiGLU (limit 10)
    with BF16 rounding after SiLU and multiplication.
    """
    if chunk not in (128, 256, 512, 1024):
        raise ValueError("Unsupported quantization chunk")
    if (
        x.ndim != 2
        or not x.is_contiguous()
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.shape[1] == 0
        or x.shape[1] % 256
    ):
        raise ValueError("Expected contiguous CUDA BF16 gate/up matrix of width 256*n")
    if padded.numel() != 1:
        raise ValueError("Expected scalar padded row count")
    for value in (sorted_ids, padded):
        if (
            value.device != x.device
            or not value.is_contiguous()
            or value.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Expected contiguous device integer metadata")
    M = x.shape[0]
    N = x.shape[1] // 2
    q = torch.empty(M, N, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(M, N // 128, device=x.device, dtype=torch.float32)
    _quant[(132,)](x, q, s, sorted_ids, padded, M, N, chunk, 132, num_warps=4, enable_fp_fusion=False)
    return q, s
