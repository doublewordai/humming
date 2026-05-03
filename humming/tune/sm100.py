from humming import dtypes
from humming.tune.sm8x import Sm80Heuristics


# TODO (mgoin): add proper heuristics
class Sm100Heuristics(Sm80Heuristics):
    max_smem_size: int = 227 * 1024
    sm_version: int = 100
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8, dtypes.float8e4m3, dtypes.float8e5m2]
