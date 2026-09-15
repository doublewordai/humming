"""Independent live-row and graph-replay checks for MoE metadata/quantization."""

import pytest
import torch

from humming.ops.moe_indices import masked_indices
from humming.ops.moe_quant import quant_valid
from humming.ops.moe_sorted_quant import quant_sorted

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(x, activation=False):
    if activation:
        gate, up = x.float().chunk(2, -1)
        gate = gate.clamp_max(10)
        up = up.clamp(-10, 10)
        silu = (gate / (1 + torch.exp(-gate))).bfloat16().float()
        x = (silu * up).bfloat16()
    values = x.float().reshape(x.shape[0], -1, 128)
    scales = (values.abs().amax(-1).clamp_min(1e-10).double() / 448).float()
    q = (values.double() / scales.double().unsqueeze(-1)).float().to(torch.float8_e4m3fn).reshape(x.shape)
    return q, scales


@pytest.mark.parametrize("block", [8, 16, 32, 64, 96, 176])
def test_indices_changing_counts_graph(block):
    counts = torch.tensor([0, 1, 17, 32], device="cuda", dtype=torch.int32)
    masked_indices(counts, 32, block)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids, experts, padded = masked_indices(counts, 32, block)
    for ns in ([0, 1, 17, 32], [32, 0, 2, 0], [0, 0, 0, 0]):
        counts.copy_(torch.tensor(ns, device="cuda", dtype=torch.int32))
        graph.replay()
        expected_ids, expected_experts = [], []
        for e, n in enumerate(ns):
            rounded = (n + block - 1) // block * block
            expected_ids.extend([e * 32 + i if i < n else 128 for i in range(rounded)])
            expected_experts.extend([e] * (rounded // block))
        assert padded.item() == len(expected_ids)
        assert ids[: len(expected_ids)].tolist() == expected_ids
        assert experts[: len(expected_experts)].tolist() == expected_experts


@pytest.mark.parametrize("kind", ["grouped_masked", "grouped_contiguous"])
@pytest.mark.parametrize("activation", [False, True])
def test_valid_rows(kind, activation):
    torch.manual_seed(317)
    ns = [0, 3, 16, 1]
    width = 512 * (2 if activation else 1)
    rows = 64 if kind == "grouped_masked" else sum(ns)
    x = (torch.randn(rows, width, device="cuda") * 3).bfloat16()
    offsets = [0]
    for n in ns:
        offsets.append(offsets[-1] + n)
    layout = torch.tensor(ns if kind == "grouped_masked" else offsets, device="cuda", dtype=torch.int32)
    live = (
        [e * 16 + i for e, n in enumerate(ns) for i in range(n)]
        if kind == "grouped_masked"
        else list(range(rows))
    )
    q, scales = quant_valid(x, layout, kind, 4, activation=activation)
    expected_q, expected_scales = reference(x[live], activation)
    if not activation:
        assert torch.equal(q.view(torch.uint8)[live], expected_q.view(torch.uint8))
        torch.testing.assert_close(scales[live], expected_scales, rtol=0, atol=0)
    else:
        # SiLU exp approximations may straddle a BF16 rounding boundary.
        torch.testing.assert_close(scales[live], expected_scales, rtol=0.01, atol=1e-7)
        actual = q.float()[live].reshape(len(live), -1, 128) * scales[live, :, None]
        expected = expected_q.float().reshape(len(live), -1, 128) * expected_scales[:, :, None]
        torch.testing.assert_close(actual, expected, rtol=0.08, atol=0.03)


def test_sorted_matches_masked_fusion():
    torch.manual_seed(901)
    x = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
    counts = torch.tensor([0, 3, 16, 1], device="cuda", dtype=torch.int32)
    ids, _, padded = masked_indices(counts, 16, 8)
    q1, s1 = quant_valid(x, counts, "grouped_masked", 4, activation=True)
    q2, s2 = quant_sorted(x, ids, padded)
    live = [e * 16 + i for e, n in enumerate([0, 3, 16, 1]) for i in range(n)]
    assert torch.equal(q1.view(torch.uint8)[live], q2.view(torch.uint8)[live])
    torch.testing.assert_close(s1[live], s2[live], rtol=0, atol=0)
