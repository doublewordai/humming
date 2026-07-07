import math
import os
from typing import TYPE_CHECKING

import numpy as np

from humming import dtypes
from humming.config import GemmType
from humming.tune.base import DeviceHeuristics

if TYPE_CHECKING:
    from humming.layer import HummingLayerMeta


class Sm90Heuristics(DeviceHeuristics):
    max_smem_size: int = 227 * 1024
    b16_allowed_dtypes: list[dtypes.DataType] = [dtypes.float16, dtypes.bfloat16]
    b8_allowed_dtypes: list[dtypes.DataType] = [
        dtypes.int8,
        dtypes.float8e4m3,
        dtypes.float8e5m2,
    ]
    b4_allowed_dtypes: list[dtypes.DataType] = []
    sm_version: int = 90

    @classmethod
    def get_config1(
        cls,
        meta: "HummingLayerMeta",
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        if use_f16_accum:
            max_block_m = 256
        elif meta.use_fused_e8m0_scale and meta.input_scale_group_size > 0:
            # The correct group-input-scale fold keeps a second C accumulator
            # live through the mainloop, so the register budget per tile is
            # roughly halved versus the plain fused-E8M0 path. BlockM 176
            # falls off a register/spill cliff (measured M192 indexed:
            # 8.37 ms/layer at [176,256,128] vs 2.12 ms at [96,256,128];
            # BlockM sweep at M96 puts the optimum at 96). Cap the tile.
            max_block_m = 96
        else:
            max_block_m = 176

        num_blocks_list = cls.calc_num_block_list(meta, shape_m, max_block_m)
        block_shape_m = np.argmin(num_blocks_list).item() * 8 + 8
        warp_shape_n = 32
        warp_shape_k = 1024 // meta.a_dtype.num_bits

        if meta.shape_n <= 4096 and not use_batch_invariant and block_shape_m <= 64:
            block_shape_n = 128
            block_shape_k = warp_shape_k * 2
            if block_shape_m <= 32:
                block_shape_k = block_shape_k * 2
            if block_shape_k > 256:
                block_shape_k = block_shape_k // 2
                warp_shape_k = warp_shape_k // 2
        else:
            block_shape_n = 256
            block_shape_k = warp_shape_k
            if block_shape_m <= 32 and meta.b_dtype.num_bits <= 6:
                block_shape_k = block_shape_k * 2
            elif block_shape_m <= 32:
                warp_shape_k = warp_shape_k // 2

        config = {
            "block_shape": (block_shape_m, block_shape_n, block_shape_k),
            "warp_shape": (block_shape_m, warp_shape_n, warp_shape_k),
            "use_stream_k": not use_batch_invariant,
            "use_f16_accum": use_f16_accum,
            "num_stages": 4,
        }

        if gemm_type != GemmType.INDEXED:
            config["use_warp_spec"] = True
            config["use_tma"] = True
            config["use_mbarrier"] = True

            if meta.shape_n % (block_shape_n * 2) == 0 and shape_m / block_shape_m >= 4:
                if gemm_type == GemmType.DENSE:
                    config["multi_cast_size_a"] = 2
        elif os.environ.get("HUMMING_INDEXED_WARP_SPEC", "1") == "1":
            # Indexed gathers can't use TMA for A, but the producer/consumer
            # split works with cp.async + mbarriers.
            config["use_warp_spec"] = True
            config["use_mbarrier"] = True

            forced_stages = int(os.environ.get("HUMMING_INDEXED_STAGES", "0"))
            if forced_stages:
                config["num_stages"] = forced_stages

            forced_n = int(os.environ.get("HUMMING_INDEXED_BLOCKN", "0"))
            if forced_n:
                config["block_shape"] = (min(block_shape_m, 64), forced_n, warp_shape_k)
                config["warp_shape"] = (min(block_shape_m, 64), 32, warp_shape_k)

            if (
                os.environ.get("HUMMING_INDEXED_PDQ", "0") == "1"
                and meta.use_fused_e8m0_scale
                and block_shape_m <= 96
            ):
                # Producer-dequant shape: one math warpgroup (BlockN 128) plus
                # one producer/dequant warpgroup; B fragments land in smem as
                # FP8 so the math warps issue no dequant instructions.
                # BlockM capped at 64: one warpgroup with the double
                # accumulator needs ~2*BlockM regs/thread for C alone, and 96
                # rows (~264 regs live) exceeds the 255-reg architectural
                # ceiling -> unavoidable spills (measured 192-1100 LDL/STL).
                pdq_block_m = min(block_shape_m, 64)
                config["block_shape"] = (pdq_block_m, 128, warp_shape_k)
                config["warp_shape"] = (pdq_block_m, 32, warp_shape_k)
                config["use_producer_dequant"] = True

        return config

    @classmethod
    def get_config2(
        cls,
        meta: "HummingLayerMeta",
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        if use_f16_accum:
            max_block_m = 256
        elif meta.input_scale_group_size > 0:
            max_block_m = 160
        elif meta.weight_scale_group_size < 128:
            max_block_m = 192
        else:
            max_block_m = 200

        num_blocks_list = cls.calc_num_block_list(meta, shape_m, max_block_m)
        block_shape_m = np.argmin(num_blocks_list).item() * 8 + 8

        block_shape_k = 256 if block_shape_m <= 32 else 128

        config = {
            "block_shape": (block_shape_m, 128, block_shape_k),
            "warp_shape": (block_shape_m, 16, 128),
            "use_stream_k": not use_batch_invariant,
            "use_f16_accum": use_f16_accum,
            "num_stages": 4,
        }

        if gemm_type != GemmType.INDEXED:
            config["use_warp_spec"] = True
            config["use_tma"] = True
            config["use_mbarrier"] = True

            if shape_m / block_shape_m >= 4 and gemm_type == GemmType.DENSE:
                config["multi_cast_size_a"] = 2
        elif os.environ.get("HUMMING_INDEXED_WARP_SPEC", "1") == "1":
            config["use_warp_spec"] = True
            config["use_mbarrier"] = True

        return config

    @classmethod
    def calc_num_block_list(
        cls,
        meta: "HummingLayerMeta",
        shape_m: int,
        max_block_m: int,
    ):
        num_blocks_list = []
        if not meta.num_experts:
            for i in range(max_block_m // 8):
                block_m = i * 8 + 8
                num_blocks_list.append(math.ceil(shape_m / block_m))
        else:
            random_state = np.random.RandomState(seed=0)
            samples = random_state.randint(0, meta.num_experts, size=shape_m)
            counts = np.bincount(samples)
            for i in range(max_block_m // 8):
                block_m = i * 8 + 8
                num_blocks = int(np.ceil(counts * 1.1 / block_m).sum().item())
                num_blocks_list.append(num_blocks)

        for i in range(max_block_m // 8):
            num_blocks = num_blocks_list[i]
            block_m = i * 8 + 8
            if meta.a_dtype == dtypes.int8 and num_blocks % 16 == 8 and block_m > 32:
                num_blocks_list[i] = 10000

        return num_blocks_list

    @classmethod
    def get_config(
        cls,
        meta: "HummingLayerMeta",
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        if meta.a_dtype.num_bits == 16:
            func = cls.get_config1
        elif meta.input_scale_group_size == 0 and meta.weight_scale_group_size == 0:
            func = cls.get_config1
        elif meta.use_fused_e8m0_scale:
            func = cls.get_config1
        else:
            func = cls.get_config2

        return func(meta, shape_m, use_f16_accum, use_batch_invariant, gemm_type)
