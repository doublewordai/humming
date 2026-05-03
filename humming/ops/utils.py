import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.utils.cpp_extension
from filelock import FileLock

import humming.utils.jit as jit_utils
from humming.utils.cuda import filter_cuda_paths

_libs = {}
_launcher_inited = False


def register_op(
    name: str,
    impl_func: Callable,
    fake_impl_func: Callable | None = None,
    mutates_args: list[str] | None = None,
):
    mutates_args = [] if mutates_args is None else mutates_args
    schema_str = torch.library.infer_schema(impl_func, mutates_args=mutates_args)
    lib_name, op_name = name.split("::")

    if lib_name not in _libs:
        _lib = torch.library.Library(lib_name, "FRAGMENT")
        _libs[lib_name] = _lib

    _lib = _libs[lib_name]
    _lib.define(op_name + schema_str)
    _lib.impl(op_name, impl_func, dispatch_key="CUDA")
    if fake_impl_func is not None:
        _lib._register_fake(op_name, fake_impl_func)


def get_humming_launcher_build_dir():
    import humming

    dirname = os.path.dirname(humming.__file__)
    launcher_code_hash = jit_utils.hash_path_content(
        path=os.path.join(dirname, "csrc/launcher/"),
        releative=True,
    )

    cache_dir = jit_utils.get_humming_cache_dir()
    py_version = f"py{sys.version_info.major}{sys.version_info.minor}"
    torch_major, torch_minor = torch.__version__.split(".")[:2]
    torch_version = f"torch{torch_major}{torch_minor}"
    version = py_version + "_" + torch_version

    launcher_build_dir = os.path.join(cache_dir, f"launcher/{version}/{launcher_code_hash}")
    Path(launcher_build_dir).mkdir(exist_ok=True, parents=True)
    return launcher_build_dir


def init_humming_launcher():
    from packaging.version import Version
    from torch.library import register_fake

    from humming.config import GemmType
    from humming.kernel import HummingKernel

    global _launcher_inited
    if _launcher_inited:
        return

    USE_TORCH_STABLE_API = Version(torch.__version__) >= Version("2.10")
    lock_filename = jit_utils.get_humming_lock_filename("launcher")
    with FileLock(lock_filename):
        import humming

        build_dir = get_humming_launcher_build_dir()
        torch_lock_file = os.path.join(build_dir, "lock")
        if os.path.exists(torch_lock_file):
            os.unlink(torch_lock_file)

        dirname = os.path.dirname(humming.__file__)
        filename = os.path.join(dirname, "csrc/launcher/launcher.cpp")

        cuda_env = filter_cuda_paths(
            required_headers=["cuda.h", "crt/host_defines.h", "cuda/std/cstdint"],
        )

        torch.utils.cpp_extension.load(
            name="humming_launcher",
            sources=[filename],
            extra_include_paths=list(cuda_env["include_paths"]),
            extra_ldflags=["-lcuda", "-lc10_cuda", "-ltorch_cuda"],
            extra_cflags=["-O3", f"-DUSE_TORCH_STABLE_API={USE_TORCH_STABLE_API}"],
            build_directory=build_dir,
        )

        _launcher_inited = True

    @register_fake("humming::launch_kernel")
    def _launch_kernel_fake(
        configs: list[int],
        inputs: torch.Tensor,
        weight: torch.Tensor,
        outputs: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
        weight_scale: torch.Tensor | None = None,
        zero_point: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        global_scale: torch.Tensor | None = None,
        sorted_ids: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
        num_tokens_padded: torch.Tensor | None = None,
        expert_layout: torch.Tensor | None = None,
        locks: torch.Tensor | None = None,
        top_k: int = 1,
        valid_shape_m: int = 0,
    ) -> torch.Tensor:
        kernel_id = configs[2]
        kernel = HummingKernel._id2kernel[kernel_id]
        shape_m = inputs.size(0)
        if kernel.gemm_type == GemmType.INDEXED:
            shape_m = inputs.size(0) * top_k
        shape_n = kernel.shape_n - kernel.pad_shape_n
        return torch.empty((shape_m, shape_n), dtype=inputs.dtype, device=inputs.device)


def build_humming_launcher_in_bg():
    if os.getenv("HUMMING_DISABLE_PARALLEL_BUILD", "0") == "1":
        return None
    cmd = "import humming.ops.utils; humming.ops.utils.init_humming_launcher()"
    env = os.environ.copy()
    env["HUMMING_DISABLE_PARALLEL_BUILD"] = "1"
    subprocess.Popen(
        [sys.executable, "-c", cmd],
        env=env,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )


build_humming_launcher_in_bg()
