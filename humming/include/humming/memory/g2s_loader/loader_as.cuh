#pragma once

#include <humming/utils/all.cuh>


template <
    class SharedStorage,
    class ProblemShape, class BlockShape, class PadShape,
    class ElementA,
    class LayerConfig, class ComputeConfig, class TuningConfig>
class G2SMemoryLoaderAS {
private:
  static constexpr bool kUseWarpSpec = TuningConfig::kUseWarpSpec;
  static constexpr bool kUseCpAsync = TuningConfig::kUseCpAsync;
  static constexpr bool kIsDenseGemm = ComputeConfig::kGemmType == GemmType::DENSE;
  static constexpr bool kIsIndexedGemm = ComputeConfig::kGemmType == GemmType::INDEXED;
  static constexpr bool kIsGroupedGemm = ComputeConfig::kGemmType == GemmType::GROUPED_CONTIGUOUS || ComputeConfig::kGemmType == GemmType::GROUPED_MASKED;

  static constexpr uint32_t kNumLoadThreads = TuningConfig::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = TuningConfig::kNumThreads - kNumLoadThreads;

  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsChannelScale = kHasInputScale && LayerConfig::kInputScaleGroupSize == 0;
  static constexpr bool kIsGroupScale = kHasInputScale && LayerConfig::kInputScaleGroupSize > 0;
  static constexpr bool kMMajorInputScale = ComputeConfig::kUseMMajorInputScale && kIsGroupScale;
  static_assert(!kMMajorInputScale || !kIsIndexedGemm);
  static constexpr bool kUseTma = TuningConfig::kUseTmaAS && kHasInputScale && !kIsIndexedGemm;
  static_assert(!kUseTma || kMMajorInputScale || kIsChannelScale);
  static constexpr uint32_t kGroupSize = kIsGroupScale ? LayerConfig::kInputScaleGroupSize : ProblemShape::K;

  static_assert(ProblemShape::K == kGroupSize || (ProblemShape::K - PadShape::K) % kGroupSize == 0);
  static constexpr uint32_t kProblemNumGroups = CEIL_DIV(ProblemShape::K - PadShape::K, kGroupSize);
  static constexpr uint32_t kNumGroups = CEIL_DIV(BlockShape::K, kGroupSize);
  static constexpr uint32_t kLoadsPerGroup = CEIL_DIV(kGroupSize, BlockShape::K);

  using LoadType = typename LoadTypeChooser<kNumGroups * 4>::Type;

public:
  SharedStorage &smem;
  const uint32_t thread_id = threadIdx.x - kLoadThreadOffset;
  const CUtensorMap *tensor_map_ptr;
  const uint32_t *gmem_ptr_raw;
  const uint32_t *gmem_ptr;

  uint32_t shape_m;
  uint32_t total_shape_m;
  uint32_t block_shape_m;
  uint32_t row_offset;
  uint32_t load_row_index;
  uint32_t col_offset = 0;
  uint32_t counter = 0;

  CUDA_INLINE
  G2SMemoryLoaderAS(const void *ptr, SharedStorage &smem, uint32_t shape_m)
      : smem(smem), shape_m(shape_m), total_shape_m((shape_m + 3u) & ~3u) {
    if constexpr (kUseTma) {
      tensor_map_ptr = reinterpret_cast<const CUtensorMap *>(ptr);
    } else {
      gmem_ptr_raw = reinterpret_cast<const uint32_t *>(ptr);
    }
  }

  template <bool kShouldAdvance = true>
  CUDA_INLINE void load(void *smem_ptr, void *mbar_ptr) {
    counter = kLoadsPerGroup != 1 ? (counter + 1) % kLoadsPerGroup : 0;
    if constexpr (kUseTma) load_tma(smem_ptr, mbar_ptr);
    else if constexpr (kMMajorInputScale) load_legacy_m_major(smem_ptr);
    else load_legacy(smem_ptr);
    if constexpr (kShouldAdvance) advance();
  }

  CUDA_INLINE void load_legacy_m_major(void *smem_ptr) {
    const int4 *gmem4 = reinterpret_cast<const int4 *>(gmem_ptr);
    int4 *smem4 = reinterpret_cast<int4 *>(smem_ptr);
    uint32_t block_m_aligned = MIN(total_shape_m - row_offset, BlockShape::M);
    PRAGMA_UNROLL
    for (uint32_t g = 0; g < kNumGroups; g++) {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < CEIL_DIV(BlockShape::M / 4, kNumLoadThreads); i++) {
        uint32_t m4 = i * kNumLoadThreads + thread_id;
        uint32_t smem_offset = (g * BlockShape::M) / 4 + m4;
        uint32_t gmem_offset = (g * total_shape_m) / 4 + m4;
        legacy_load_pred<kUseCpAsync>(gmem4 + gmem_offset, smem4 + smem_offset, m4 * 4 < block_m_aligned);
      }
    }
  }

  CUDA_INLINE void load_tma(void *smem_ptr, void *mbar_ptr) {
    if (thread_id == 0) tma_load_2d(tensor_map_ptr, smem_ptr, mbar_ptr, row_offset, col_offset);
  }

  CUDA_INLINE void load_legacy(void *smem_ptr) {
    if constexpr (!kIsIndexedGemm && kIsChannelScale) {
      uint32_t *smem_ptr_load = reinterpret_cast<uint32_t *>(smem_ptr);
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < CEIL_DIV(BlockShape::M, kNumLoadThreads); i++) {
        uint32_t idx = i * kNumLoadThreads + thread_id;
        legacy_load_pred<kUseCpAsync>(gmem_ptr + idx, smem_ptr_load + idx, idx < block_shape_m);
      }
    } else {
      constexpr uint32_t kSmemStride = kNumGroups / (sizeof(LoadType) / 4);
      constexpr uint32_t kGmemStride = kProblemNumGroups / (sizeof(LoadType) / 4);

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < CEIL_DIV(BlockShape::M, kNumLoadThreads); i++) {
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < kSmemStride; j++) {
          uint32_t smem_offset = (i * kNumLoadThreads + thread_id) * kSmemStride + j;
          uint32_t smem_row = smem_offset / kSmemStride;
          uint32_t smem_col = smem_offset % kSmemStride;

          uint32_t gmem_row = kIsIndexedGemm ? load_row_index : smem_row;
          uint32_t gmem_offset = gmem_row * kGmemStride + smem_col;

          const LoadType *gmem_ptr_load = reinterpret_cast<const LoadType *>(gmem_ptr);
          LoadType *smem_ptr_load = reinterpret_cast<LoadType *>(smem_ptr);
          bool pred = kIsIndexedGemm ? (gmem_row < shape_m) : (smem_row < block_shape_m);
          legacy_load_pred<kUseCpAsync>(gmem_ptr_load + gmem_offset, smem_ptr_load + smem_offset, pred);
        }
      }
    }
  }

  CUDA_INLINE
  void advance() {
    if (kIsGroupScale && (kLoadsPerGroup == 1 || counter == 0)) {
      col_offset += kNumGroups;
      if constexpr (!kUseTma && kMMajorInputScale) {
        gmem_ptr += kNumGroups * total_shape_m;
      } else if constexpr (!kUseTma) {
        gmem_ptr += kNumGroups;
      }
    }
  }

  CUDA_INLINE
  void seek(uint32_t m_block_id, uint32_t k_block_id, uint32_t current_shape_m, uint32_t m_offset) {
    if constexpr (kIsGroupScale) {
      if constexpr (BlockShape::K >= kGroupSize) {
        col_offset = k_block_id * kNumGroups;
      } else {
        col_offset = (k_block_id * BlockShape::K) / kGroupSize;
      }
    } else {
      col_offset = 0;
    }

    if constexpr (kIsGroupedGemm) {
      shape_m = current_shape_m;
      row_offset = m_offset;
    } else {
      row_offset = m_block_id * BlockShape::M;
    }
    block_shape_m = MIN((shape_m - row_offset), BlockShape::M);
    if constexpr (kUseTma) {} else if constexpr (kMMajorInputScale) {
      gmem_ptr = gmem_ptr_raw + (col_offset * total_shape_m + row_offset);
    } else if constexpr (!kIsIndexedGemm) {
      gmem_ptr = gmem_ptr_raw + ((row_offset * kProblemNumGroups) + col_offset);
    } else {
      gmem_ptr = gmem_ptr_raw + col_offset;

      constexpr uint32_t kSmemStride = kNumGroups / (sizeof(LoadType) / 4);
      uint32_t smem_row = thread_id / kSmemStride;

      if (smem_row < BlockShape::M) {
        load_row_index = smem.rd_row_index[smem_row];
      } else {
        load_row_index = shape_m;
      }
    }
  }
};
