"""MXFP4 W4A8 coverage with grouped FP8 inputs."""

import pytest

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig, MmaType
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)

SHAPE_N = 1024
SHAPE_K = 1024
INPUT_GROUP_SIZE = 128
WEIGHT_GROUP_SIZE = 32
NUM_EXPERTS = 8


def _layer_config(
    *,
    num_experts: int = 0,
    use_fused_e8m0_scale: bool | None = None,
) -> LayerConfig:
    return LayerConfig(
        shape_n=SHAPE_N,
        shape_k=SHAPE_K,
        num_experts=num_experts,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e8m0,
        input_scale_group_size=INPUT_GROUP_SIZE,
        weight_scale_group_size=WEIGHT_GROUP_SIZE,
        mma_type=MmaType.WGMMA,
        use_fused_e8m0_scale=use_fused_e8m0_scale,
    )


def _case(
    name: str,
    *,
    gemm_type: GemmType = GemmType.DENSE,
    use_fused_e8m0_scale: bool | None = None,
) -> KernelTestCase:
    is_dense = gemm_type == GemmType.DENSE
    return KernelTestCase(
        name=name,
        layer_config=_layer_config(
            num_experts=0 if is_dense else NUM_EXPERTS,
            use_fused_e8m0_scale=use_fused_e8m0_scale,
        ),
        compute_config=ComputeConfig(gemm_type=gemm_type),
        top_k=1 if is_dense else 2,
        seed=2026,
    )


MXFP4_CASES = (
    (True, _case("mxfp4-grouped-fp8-dense-auto")),
    (False, _case("mxfp4-grouped-fp8-dense-nonfused", use_fused_e8m0_scale=False)),
    (True, _case("mxfp4-grouped-fp8-indexed-auto", gemm_type=GemmType.INDEXED)),
    (True, _case("mxfp4-grouped-fp8-grouped-masked-auto", gemm_type=GemmType.GROUPED_MASKED)),
)


@pytest.mark.parametrize(
    "expected_fused,test_case",
    MXFP4_CASES,
    ids=[case.name for _, case in MXFP4_CASES],
)
def test_mxfp4(expected_fused, test_case):
    config = test_case.layer_config
    assert config.use_fused_e8m0_scale is expected_fused
    assert config.is_group_weight_scale
    assert config.is_tensor_weight_scale_2 is expected_fused

    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=config.mma_type.value)
    results = KernelTestRunner(test_case).run()
    assert_kernel_test_shape_coverage(results)


def test_mxfp4_case_coverage():
    assert {expected_fused for expected_fused, _ in MXFP4_CASES} == {False, True}
    assert {case.compute_config.gemm_type for _, case in MXFP4_CASES} == {
        GemmType.DENSE,
        GemmType.INDEXED,
        GemmType.GROUPED_MASKED,
    }
    assert all(case.layer_config.input_scale_group_size > 0 for _, case in MXFP4_CASES)


@pytest.mark.parametrize("block_m", [8, 16, 32])
@pytest.mark.parametrize("stream_k", [False, True])
def test_indexed_fp32_accumulator_access(block_m, stream_k, monkeypatch):
    """Small indexed tiles exercise local accumulator spills and smem reduction.

    Run under compute-sanitizer as well as numerical comparison: the original
    vector local accesses can fault even when the GEMM shape is supported.
    """
    from humming.tune import get_heuristics_config

    test_case = _case("indexed-accumulator-access", gemm_type=GemmType.INDEXED)
    skip_if_unsupported(a_dtype=test_case.layer_config.a_dtype, mma_type="wgmma")
    config = dict(get_heuristics_config(test_case.layer_config, shape_m=64, gemm_type="indexed"))
    _, block_n, _ = config["block_shape"]
    _, warp_n, warp_k = config["warp_shape"]
    config.update(
        block_shape=(block_m, block_n, 256),
        warp_shape=(block_m, warp_n, warp_k),
        use_stream_k=stream_k,
    )
    monkeypatch.setattr(
        "humming.testing.runner.generate_heuristics_configs",
        lambda layer, compute, shapes: [dict(config) for _ in shapes],
    )
    results = KernelTestRunner(test_case).run((1, 17, 257))
    assert_kernel_test_shape_coverage(results, (1, 17, 257))
