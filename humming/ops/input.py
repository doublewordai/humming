import torch
import triton
import triton.language as tl
from torch._subclasses.fake_tensor import FakeTensor


@triton.jit
def calc_scale(tensor, dtype):
    if dtype == "float8e4m3":
        absmax = tl.maximum(tl.max(tl.abs(tensor)), 1e-30)
        scale_x = absmax / 448
    elif dtype == "float8e5m2":
        absmax = tl.maximum(tl.max(tl.abs(tensor)), 1e-30)
        scale_x = absmax / 57344
    elif dtype == "float4e2m1":
        absmax = tl.maximum(tl.max(tl.abs(tensor)), 1e-30)
        scale_x = absmax / 6
    elif dtype == "int8":
        maxval = tl.maximum(tl.max(tensor), 1e-30)
        minval = tl.minimum(tl.min(tensor), -1e-30)
        scale_x = tl.maximum(maxval / 127.49, -minval / 128.49)
    elif dtype == "int4":
        maxval = tl.maximum(tl.max(tensor), 1e-30)
        minval = tl.minimum(tl.min(tensor), -1e-30)
        scale_x = tl.maximum(maxval / 7.49, -minval / 8.49)
    else:
        tl.static_assert(False, "unsupported dtype: " + dtype)
    return scale_x


@triton.jit
def quant_tensor(tensor, dtype):
    if dtype == "float8e4m3":
        tensor = tensor.to(tl.float8e4nv)
    elif dtype == "float8e5m2":
        tensor = tensor.to(tl.float8e5)
    elif dtype == "int8":
        tensor = tl.inline_asm_elementwise(
            asm="cvt.rni.s8.f32 $0, $1;",
            constraints="=r,f",
            args=[tensor],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        tensor = tensor.to(tl.int8)
    else:
        tl.static_assert(False, "unsupported dtype: " + dtype)

    return tensor


@triton.jit
def quant_tensor_x2(tensor1, tensor2, dtype):
    if dtype == "int4":
        tensor = tl.inline_asm_elementwise(
            asm="""{
            .reg .s32 r1, r2, a1, a2;
            cvt.rni.s32.f32 r1, $1;
            cvt.rni.s32.f32 r2, $2;
            and.b32 a1, r1, 0xF;
            and.b32 a2, r2, 0xF;
            mad.lo.s32 $0, a2, 16, a1;
            }""",
            constraints="=r,f,f",
            args=[tensor1, tensor2],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        tensor = tensor.to(tl.uint8)
    elif dtype == "float4e2m1":
        tensor = tl.inline_asm_elementwise(
            asm="cvt.rn.satfinite.e2m1x2.f32 $0, $1, $2;",
            constraints="=r,f,f",
            args=[tensor1, tensor2],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        tensor = tensor.to(tl.uint8)
    else:
        tl.static_assert(False, "unsupported dtype: " + dtype)

    return tensor


@triton.jit
def _quant_tensor_kernel(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    num_blocks,
    is_dynamic: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    dtype: tl.constexpr,
):
    block_id = tl.program_id(0).to(tl.int64)
    tl.static_assert(N % GROUP_SIZE == 0)

    row_num_blocks = N // GROUP_SIZE

    for g in tl.static_range(GROUPS_PER_BLOCK):
        group_id = block_id * GROUPS_PER_BLOCK + g
        in_range = group_id < num_blocks
        row_id = group_id // row_num_blocks
        col_block_id = group_id % row_num_blocks
        offset = row_id * stride_x + col_block_id * GROUP_SIZE

        if dtype == "int4" or dtype == "float4e2m1":
            cols = tl.arange(0, BLOCK // 2)
            mask = (cols < GROUP_SIZE // 2) & in_range
            cols1 = cols * 2
            cols2 = cols * 2 + 1

            x1 = tl.load(x_ptr + (offset + cols1), mask=mask, other=0.0).to(tl.float32)
            x2 = tl.load(x_ptr + (offset + cols2), mask=mask, other=0.0).to(tl.float32)

            scale = (
                tl.maximum(calc_scale(x1, dtype), calc_scale(x2, dtype))
                if is_dynamic
                else tl.load(scale_ptr + col_block_id, mask=in_range)
            )
            inv_scale = 1 / scale
            x_q = quant_tensor_x2(x1 * inv_scale, x2 * inv_scale, dtype)
            tl.store(xq_ptr + group_id * GROUP_SIZE // 2 + cols, x_q, mask=mask)

            if is_dynamic:
                tl.store(scale_ptr + row_id * row_num_blocks + col_block_id, scale, mask=in_range)
        else:
            cols = tl.arange(0, BLOCK)
            mask = (cols < GROUP_SIZE) & in_range
            x = tl.load(x_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
            scale = (
                calc_scale(x, dtype)
                if is_dynamic
                else tl.load(scale_ptr + col_block_id, mask=in_range)
            )
            inv_scale = 1 / scale
            x_q = quant_tensor(x * inv_scale, dtype)
            tl.store(xq_ptr + group_id * GROUP_SIZE + cols, x_q, mask=mask)

            if is_dynamic:
                tl.store(scale_ptr + row_id * row_num_blocks + col_block_id, scale, mask=in_range)


def quant_input(
    inputs: torch.Tensor,
    dtype: str,
    scales: torch.Tensor | None = None,
    outputs: torch.Tensor | None = None,
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dtype in ["int4", "float4e2m1"]:
        output_shape = (*inputs.shape[:-1], inputs.size(-1) // 2)
        output_dtype = torch.uint8
    elif dtype == "int8":
        output_shape = inputs.shape
        output_dtype = torch.int8
    elif dtype == "float8e4m3":
        output_shape = inputs.shape
        output_dtype = torch.float8_e4m3fn
    elif dtype == "float8e5m2":
        output_shape = inputs.shape
        output_dtype = torch.float8_e5m2
    else:
        raise ValueError("unsupported dtype: " + dtype)

    if outputs is None:
        outputs = torch.empty(output_shape, dtype=output_dtype, device=inputs.device)

    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert outputs.shape == output_shape
    assert outputs.dtype == output_dtype
    assert outputs.device == inputs.device

    is_dynamic = scales is None
    inputs = inputs.view(-1, inputs.size(-1))
    if group_size is None or group_size == 0:
        group_size = inputs.size(1)
    assert inputs.size(1) % group_size == 0
    num_blocks = inputs.nelement() // group_size

    if is_dynamic:
        scales = torch.empty(
            (inputs.size(0), inputs.size(1) // group_size),
            dtype=torch.float32,
            device=inputs.device,
        )

    if not isinstance(inputs, FakeTensor):
        assert inputs.is_cuda
        BLOCK = triton.next_power_of_2(group_size)
        # Merge multiple groups per block to reduce scheduling overhead.
        groups_per_block = 1
        if group_size <= 256 and num_blocks >= 131072:
            # Small group_size (e.g. 128) with massive block count
            groups_per_block = min(1024 // group_size, num_blocks)
        grid_blocks = (num_blocks + groups_per_block - 1) // groups_per_block
        effective_block = BLOCK // 2 if dtype in ("int4", "float4e2m1") else BLOCK
        num_warps = min(max(effective_block // 256, 1), 8)
        num_stages = 1

        _quant_tensor_kernel[(grid_blocks,)](
            inputs,
            outputs,
            scales,
            inputs.stride(0),
            num_blocks,
            is_dynamic,
            inputs.size(1),
            group_size,
            BLOCK,
            groups_per_block,
            dtype,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    if scales is None:
        scales = torch.empty(0)
    else:
        scales = scales.view(*outputs.shape[:-1], scales.size(-1))

    return outputs, scales
