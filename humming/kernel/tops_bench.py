import ctypes
import dataclasses
from typing import ClassVar

import cuda.bindings.driver as cbd
import jinja2
import torch

from humming import dtypes
from humming.config import MmaOpClass, MmaType
from humming.jit.runtime import KernelRuntime

CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/tops_bench.cuh>

class MmaOpClass {
public:
{{mma_op_class}}
};

""")


@dataclasses.dataclass(kw_only=True)
class TopsBenchKernel(KernelRuntime):
    name: ClassVar[str] = "tops_bench"
    mma_type: str | MmaType
    mma_shape_m: int
    mma_shape_n: int
    mma_shape_k: int
    ab_dtype: str | dtypes.DataType
    cd_dtype: str | dtypes.DataType
    repeat_count: int
    unroll_count: int
    use_desc: bool = False
    wait_every: int = 1

    def init_kernel(self):
        if isinstance(self.mma_type, str):
            self.mma_type = MmaType(self.mma_type)
        if isinstance(self.ab_dtype, str):
            self.ab_dtype = dtypes.DataType.from_str(self.ab_dtype)
        if isinstance(self.cd_dtype, str):
            self.cd_dtype = dtypes.DataType.from_str(self.cd_dtype)

        self.mma_op_class = MmaOpClass.from_config(
            self.mma_type,
            self.mma_shape_m,
            self.mma_shape_n,
            self.mma_shape_k,
            self.ab_dtype,
            self.ab_dtype,
            self.cd_dtype,
        )

        self.code = CODE_TEMPLATE.render(
            mma_op_class=self.mma_op_class.to_cpp_str(),
            repeat_count=self.repeat_count,
            unroll_count=self.unroll_count,
        )
        if self.use_desc:
            self.kernel_expr = f"tops_bench_dd<MmaOpClass, {self.repeat_count}, {self.unroll_count}, {self.wait_every}>"
        else:
            self.kernel_expr = f"tops_bench<MmaOpClass, {self.repeat_count}, {self.unroll_count}>"
        self.arg_types = (ctypes.c_void_p,)
        self.prepare()

        self.num_warps = 32
        self.ops_per_mma_per_warp = self.mma_shape_m * self.mma_shape_n * self.mma_shape_k * 2
        if self.mma_type == MmaType.WGMMA:
            self.ops_per_mma_per_warp = self.ops_per_mma_per_warp // 4
            self.num_warps = self.num_warps // 4
        self.sm_count = torch.cuda.get_device_properties().multi_processor_count
        self.num_ctas = self.sm_count * 2
        self.ops_per_call = self.ops_per_mma_per_warp * self.num_warps * self.num_ctas

    def __call__(self):
        self.check_context()
        config = cbd.CUlaunchConfig()
        config.gridDimX = self.num_ctas
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = self.num_warps * 32
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream().cuda_stream

        tensor = torch.empty((1,), dtype=torch.uint32, device="cuda:0")
        arg_values = (tensor.data_ptr(),)

        cbd.cuLaunchKernelEx(config, self.kernel, (arg_values, self.arg_types), 0)
