#pragma once

#include <humming/utils/all.cuh>


template <
    class SharedStorage,
    class ProblemShape, class BlockShape, class PadShape,
    class ElementA,
    class LayerConfig, class ComputeConfig, class TuningConfig>
class G2SMemoryLoaderAS {
private:
  static constexpr bool kUseMxmma = LayerConfig::kMmaType == MmaType::MXMMA;
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
  static constexpr uint32_t kGroupSize = kIsGroupScale ? LayerConfig::kInputScaleGroupSize : ProblemShape::K;

  static_assert(ProblemShape::K == kGroupSize || (ProblemShape::K - PadShape::K) % kGroupSize == 0);
  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kProblemNumGroups = CEIL_DIV(ProblemShape::K - PadShape::K, kGroupSize);
  static constexpr uint32_t kNumGroups = CEIL_DIV(BlockShape::K, kGroupSize);
  static constexpr uint32_t kMxScaleVec = kPartMmaShapeK / kGroupSize;
  static constexpr uint32_t kLoadsPerGroup = kUseMxmma ? MAX(1u, 4 / kNumGroups) : CEIL_DIV(kGroupSize, BlockShape::K);

  using LoadType = typename LoadTypeChooser<kNumGroups * 4>::Type;

public:
  SharedStorage &smem;
  const uint32_t thread_id = threadIdx.x - kLoadThreadOffset;
  const CUtensorMap *tensor_map_ptr;
  const uint32_t *gmem_ptr_raw;
  const uint32_t *gmem_ptr;

  uint32_t shape_m;
  uint32_t block_shape_m;
  uint32_t row_offset;
  uint32_t load_row_index;
  uint32_t col_offset = 0;
  uint32_t counter = 0;
  const uint8_t *mx_gmem_ptr;

  CUDA_INLINE
  G2SMemoryLoaderAS(const void *ptr, SharedStorage &smem, uint32_t shape_m)
      : smem(smem), shape_m(shape_m) {
    gmem_ptr_raw = reinterpret_cast<const uint32_t *>(ptr);
  }

  template <bool kShouldAdvance = true>
  CUDA_INLINE void load(void *smem_ptr, void *mbar_ptr) {
    counter = kLoadsPerGroup != 1 ? (counter + 1) % kLoadsPerGroup : 0;
    if constexpr (kUseMxmma) {
      load_mx_legacy(smem_ptr);
    } else {
      load_legacy(smem_ptr);
    }
    if constexpr (kShouldAdvance) advance();
  }

  CUDA_INLINE void load_mx_legacy(void *smem_ptr) {
    uint32_t *smem_ptr_load = reinterpret_cast<uint32_t *>(smem_ptr);
    const uint32_t *gmem_ptr_load = reinterpret_cast<const uint32_t *>(gmem_ptr);

    constexpr uint32_t kNumRows = CEIL_DIV(BlockShape::K / kPartMmaShapeK * kMxScaleVec, 4);
    constexpr uint32_t kMxGmemStride = ProblemShape::K / kPartMmaShapeK * kMxScaleVec / 4;
    constexpr uint32_t kNumInts = BlockShape::M * kNumRows;

    if constexpr (kNumInts <= kNumLoadThreads) {
      uint32_t smem_offset = thread_id;
      uint32_t smem_row = smem_offset / BlockShape::M;
      uint32_t smem_col = smem_offset % BlockShape::M;

      uint32_t gmem_row = smem_col;
      uint32_t gmem_col = smem_row;
      uint32_t gmem_offset = gmem_row * kMxGmemStride + gmem_col;
      uint32_t pred = thread_id < kNumInts;

      legacy_load_pred<kUseCpAsync>(gmem_ptr_load + gmem_offset, smem_ptr_load + smem_offset, pred);
    } else {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kNumRows; i++) {
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < CEIL_DIV(BlockShape::M, kNumLoadThreads); j++) {
          uint32_t m_index = j * kNumLoadThreads + thread_id;
          uint32_t gmem_offset = m_index * kMxGmemStride + i;
          uint32_t smem_offset = i * BlockShape::M + m_index;
          uint32_t pred = m_index < block_shape_m;

          legacy_load_pred<kUseCpAsync>(gmem_ptr_load + gmem_offset, smem_ptr_load + smem_offset, pred);
        }
      }
    }
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
      gmem_ptr += kUseMxmma ? CEIL_DIV(kNumGroups, 4) : kNumGroups;
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
    if constexpr (!kIsIndexedGemm) {
      uint32_t k_word_offset = kUseMxmma ? col_offset / 4 : col_offset;
      uint32_t row_stride = kUseMxmma ? kProblemNumGroups / 4 : kProblemNumGroups;
      gmem_ptr = gmem_ptr_raw + ((row_offset * row_stride) + k_word_offset);
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
