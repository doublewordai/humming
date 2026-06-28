import ctypes
import dataclasses
from typing import ClassVar

import cuda.bindings.driver as cbd
import jinja2
import torch

from humming import dtypes
from humming.jit.runtime import KernelRuntime
from humming.kernel.hadamard import _pick_launch_params, _TORCH_TO_CPP_TYPE

CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/hadamard_quant.cuh>
""")


@dataclasses.dataclass(kw_only=True)
class HadamardQuantInputKernel(KernelRuntime):
    name: ClassVar[str] = "hadamard_quant_input"
    source_torch_dtype: torch.dtype
    target_dtype: dtypes.DataType
    block_size: int
    group_size: int
    has_extra_scale: bool = False
    m_major: bool = False

    def init_kernel(self):
        assert self.block_size % self.group_size == 0, (
            "group_size must divide block_size"
        )
        cpp_source = _TORCH_TO_CPP_TYPE[self.source_torch_dtype]
        cpp_target = self.target_dtype.to_cpp_str()
        threads_per_tile, tiles_per_block = _pick_launch_params(self.block_size)
        # Cross-warp reduction relies on block-wide __syncthreads, which would
        # corrupt cross-tile state if multiple tiles shared a block. Force one
        # tile per block whenever any group spans more than one warp.
        elems_per_thread = self.block_size // threads_per_tile
        lanes_per_group = self.group_size // elems_per_thread
        if lanes_per_group > 32 and tiles_per_block != 1:
            tiles_per_block = 1
        self.threads_per_tile = threads_per_tile
        self.tiles_per_block = tiles_per_block
        self.threads_per_block = threads_per_tile * tiles_per_block

        E = self.block_size // threads_per_tile
        assert self.group_size >= E and self.group_size % E == 0, (
            f"group_size {self.group_size} must be >= and a multiple of "
            f"elems_per_thread {E} (block_size={self.block_size}, T={threads_per_tile})"
        )

        self.code = CODE_TEMPLATE.render()
        self.kernel_expr = (
            f"hadamard_quant_input<\n"
            f"    {cpp_source},\n"
            f"    {cpp_target},\n"
            f"    {self.block_size},\n"
            f"    {self.group_size},\n"
            f"    {threads_per_tile},\n"
            f"    {tiles_per_block},\n"
            f"    {int(self.has_extra_scale)},\n"
            f"    {int(self.m_major)}>"
        )
        self.arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        )
        self.prepare()

    def __call__(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        scales: torch.Tensor,
        extra_scale: float = 1.0,
    ):
        self.check_context()
        assert inputs.is_contiguous() and outputs.is_contiguous() and scales.is_contiguous()
        assert inputs.size(-1) % self.block_size == 0
        assert inputs.dtype == self.source_torch_dtype
        assert scales.dtype == torch.float32

        num_tiles = inputs.numel() // self.block_size
        shape_m = inputs.numel() // inputs.size(-1)
        if self.m_major:
            shape_m = (shape_m + 3) // 4 * 4
        groups_per_row = inputs.size(-1) // self.group_size
        device = inputs.device

        config = cbd.CUlaunchConfig()
        config.gridDimX = (num_tiles + self.tiles_per_block - 1) // self.tiles_per_block
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = self.threads_per_block
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(device).cuda_stream

        arg_values = (
            inputs.data_ptr(),
            outputs.data_ptr(),
            scales.data_ptr(),
            float(extra_scale),
            num_tiles,
            shape_m,
            groups_per_row,
        )

        cbd.cuLaunchKernelEx(config, self.kernel, (arg_values, self.arg_types), 0)
