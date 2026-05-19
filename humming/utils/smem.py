import math
from typing import TYPE_CHECKING

from humming import dtypes
from humming.config import GemmType

if TYPE_CHECKING:
    from humming.layer import HummingLayerMeta


def _align_up(size: int, alignment: int = 128) -> int:
    if size == 0:
        return 0
    return math.ceil(size / alignment) * alignment


def estimate_smem_size(
    a_dtype: dtypes.DataType,
    b_dtype: dtypes.DataType,
    bs_dtype: dtypes.DataType,
    input_scale_group_size: int,
    weight_scale_group_size: int,
    has_zero_point: bool,
    is_fp_zero_point: bool,
    has_bias: bool,
    num_experts: int,
    gemm_type: GemmType,
    block_shape: tuple[int, int, int],
    num_stages: int,
):
    block_m, block_n, block_k = block_shape
    a_stage_size = a_dtype.num_bits * block_m * block_k // 8
    b_stage_size = b_dtype.num_bits * block_n * block_k // 8
    zp_num_bits = math.ceil(b_dtype.num_bits / 4) * 4
    zp_num_bits = 16 if is_fp_zero_point else zp_num_bits
    bias_size = block_n * 2 if has_bias else 0

    as_stage_size = 0
    bs_stage_size = 0
    bzp_stage_size = 0
    as_channel_size = 0
    bs_channel_size = 0
    bzp_channel_size = 0
    if weight_scale_group_size > 0:
        bs_stage_num_groups = math.ceil(block_k / weight_scale_group_size)
        bs_stage_size = block_n * bs_stage_num_groups * bs_dtype.num_bits // 8
        bzp_stage_size = block_n * bs_stage_num_groups * zp_num_bits // 8
    else:
        bs_channel_size = block_n * bs_dtype.num_bits // 8
        bzp_channel_size = block_n * zp_num_bits // 8

    if not has_zero_point:
        bzp_stage_size = 0
        bzp_channel_size = 0

    if a_dtype.num_bits < 16:
        if input_scale_group_size > 0:
            as_stage_num_groups = math.ceil(block_k / input_scale_group_size)
            as_stage_size = block_m * as_stage_num_groups * 4
        else:
            as_channel_size = block_m * 4

    offset = 0

    def add_field(size: int, alignment: int = 1):
        nonlocal offset
        if size == 0:
            return
        offset = _align_up(offset, alignment)
        offset += size

    add_field(_align_up(a_stage_size) * num_stages, 128)
    add_field(_align_up(b_stage_size) * num_stages, 128)
    add_field(as_stage_size * num_stages)
    add_field(_align_up(bs_stage_size) * num_stages, 128)
    bzp_stages = 1 if weight_scale_group_size == 0 else num_stages
    add_field(_align_up(bzp_stage_size or bzp_channel_size) * bzp_stages, 128)
    add_field(bs_channel_size, 128)
    add_field(bias_size, 128)
    add_field(as_channel_size)

    moe_tensor_size = 0
    if gemm_type == GemmType.INDEXED:
        moe_tensor_size = block_m * 4 * 2
    elif gemm_type == GemmType.GROUPED_CONTIGUOUS:
        moe_tensor_size = num_experts * 4 * 2
    elif gemm_type == GemmType.GROUPED_MASKED:
        moe_tensor_size = num_experts * 4
    size = offset + moe_tensor_size

    return size


def estimate_smem_size_layer(
    meta: "HummingLayerMeta",
    block_shape: tuple[int, int, int],
    gemm_type: GemmType,
    num_stages: int,
):
    return estimate_smem_size(
        a_dtype=meta.a_dtype,
        b_dtype=meta.b_dtype,
        bs_dtype=meta.bs_dtype or meta.c_dtype,
        input_scale_group_size=meta.input_scale_group_size,
        weight_scale_group_size=meta.weight_scale_group_size,
        has_zero_point=meta.has_zero_point,
        is_fp_zero_point=meta.is_fp_zero_point,
        has_bias=meta.has_bias,
        num_experts=meta.num_experts,
        gemm_type=gemm_type,
        block_shape=block_shape,
        num_stages=num_stages,
    )
