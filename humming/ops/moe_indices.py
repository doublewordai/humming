"""Build indexed GEMM metadata from GPU-resident masked expert counts."""

import torch
import triton
import triton.language as tl


@triton.jit
def _indices(
    C, S, E, P, M: tl.constexpr, G: tl.constexpr, B: tl.constexpr, W: tl.constexpr, NG: tl.constexpr
):
    e = tl.program_id(0)
    worker = tl.program_id(1)
    es = tl.arange(0, NG)
    ns = tl.load(C + es, es < G, 0)
    pads = tl.cdiv(ns, B) * B
    off = tl.sum(tl.where(es < e, pads, 0))
    if e == 0 and worker == 0:
        tl.store(P, tl.sum(pads))
    n = tl.load(C + e)
    col = tl.arange(0, B)
    for tile in range(worker, tl.cdiv(n, B), W):
        row = tile * B + col
        tl.store(S + off + tile * B + col, tl.where(row < n, e * M + row, G * M))
        tl.store(E + off // B + tile, e)


def masked_indices(counts, capacity, block):
    """Return sorted rows, tile experts and device padded count.

    Each count must be in [0, capacity]. No host count readback is performed.
    Entries beyond the returned padded count are uninitialized.
    """
    if (
        counts.ndim != 1
        or not counts.is_contiguous()
        or not counts.is_cuda
        or counts.dtype not in (torch.int32, torch.int64)
        or counts.numel() == 0
    ):
        raise ValueError("Expected a nonempty contiguous CUDA integer count vector")
    if capacity < 0 or block <= 0 or block & (block - 1):
        raise ValueError("Expected nonnegative capacity and power-of-two block")
    if counts.numel() * (capacity + block - 1) >= 2**31:
        raise ValueError("Indexed metadata exceeds int32 range")
    G = counts.numel()
    size = G * triton.cdiv(capacity, block) * block
    sorted_ids = torch.empty(size, device=counts.device, dtype=torch.int32)
    expert_ids = torch.empty(size // block, device=counts.device, dtype=torch.int32)
    padded = torch.empty(1, device=counts.device, dtype=torch.int32)
    _indices[(G, 4)](counts, sorted_ids, expert_ids, padded, capacity, G, block, 4, triton.next_power_of_2(G))
    return sorted_ids, expert_ids, padded
