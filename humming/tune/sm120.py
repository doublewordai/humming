from humming import dtypes
from humming.tune.sm8x import Sm89Heuristics


class Sm120Heuristics(Sm89Heuristics):
    sm_version: int = 120
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8, dtypes.float8e4m3, dtypes.float8e5m2]
