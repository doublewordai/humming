#pragma once

#include <humming/datatype/dequant_fused.cuh>
#include <humming/datatype/dequant_prepare.cuh>
#include <humming/datatype/dequant_single.cuh>
#include <humming/utils/all.cuh>


template <class SourceType, class TargetType, bool kHasZeroPoint, bool kIsFpZeroPoint>
CUDA_INLINE void dequant_b1248(const uint32_t *qb, uint32_t *res, uint32_t j, uint32_t *zp_vals = nullptr) {
  static_assert(SourceType::kBits <= TargetType::kBits);
  constexpr uint32_t kResultPackedNums = 32 / TargetType::kBits;
  constexpr uint32_t kResultPackedNumSourceBits = kResultPackedNums * SourceType::kBits;
  constexpr bool kIsFpToFp = SourceType::kIsFloatingPointType && TargetType::kIsFloatingPointType;
  constexpr uint32_t reverse_pattern = TargetType::kBits / SourceType::kBits;

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 4; i++) {
    uint32_t index = j * 4 + i;
    uint32_t zp_val = zp_vals[i];
    uint32_t qb_index = kResultPackedNumSourceBits * index / 32;

    uint32_t shift_count;
    if constexpr (SourceType::kBits * 4 % TargetType::kBits == 0) {
      shift_count = SourceType::kBits * i % TargetType::kBits;
    } else {
      shift_count = SourceType::kBits * index % TargetType::kBits;
    }

    uint32_t qb_val = qb[qb_index];
    if (kIsFpToFp && shift_count) qb_val = qb_val << shift_count;
    if (!kIsFpToFp && shift_count) qb_val = qb_val >> shift_count;
    if constexpr (!kIsFpToFp) {
      res[i] = dequant_single<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint>(qb_val, zp_val);
    } else {
      uint32_t reversed_index = i / reverse_pattern * reverse_pattern + reverse_pattern - 1 - i % reverse_pattern;
      res[reversed_index] = dequant_single<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint>(qb_val, zp_val);
    }
  }
}


template <class SourceType>
CUDA_INLINE void repack_native_mxf8f6f4(const uint32_t *qb, uint32_t *res, uint32_t j) {
  constexpr uint32_t kSrcBits = SourceType::kBits;
  constexpr uint32_t kTgtBits = 8;
  constexpr uint32_t kPaddedSrcBits = static_next_power_of_2(kSrcBits);  // 4 or 8
  constexpr uint32_t kKeepMask = ((1u << kSrcBits) - 1u) << (kTgtBits - kSrcBits);
  constexpr uint32_t kKeepMaskRep = kKeepMask * 0x01010101u;
  constexpr uint32_t kHighPad = 2;
  constexpr uint32_t reverse_pattern = kTgtBits / kPaddedSrcBits;

  if constexpr (kSrcBits == static_next_power_of_2(kSrcBits)) {
    constexpr uint32_t kResultPackedNumSourceBits = (32 / kTgtBits) * kSrcBits;
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 4; i++) {
      uint32_t index = j * 4 + i;
      uint32_t qb_index = kResultPackedNumSourceBits * index / 32;
      uint32_t shift_count = (kSrcBits * 4 % kTgtBits == 0) ? (kSrcBits * i % kTgtBits)
                                                            : (kSrcBits * index % kTgtBits);
      uint32_t qb_val = qb[qb_index];
      if (shift_count) qb_val = qb_val << shift_count;
      uint32_t reversed_index = i / reverse_pattern * reverse_pattern + reverse_pattern - 1 - i % reverse_pattern;
      res[reversed_index] = (qb_val & kKeepMaskRep) >> kHighPad;
    }
  } else {
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 4; i++) {
      uint32_t index = j * 4 + i;
      uint32_t qb_val = 0;
      if (index * kPaddedSrcBits % kTgtBits == 0) {
        const uint32_t idx1 = index / kTgtBits;
        const uint32_t idx2 = index * kPaddedSrcBits / kTgtBits % kPaddedSrcBits;
        const uint32_t qb_offset = idx1 * kSrcBits;
        qb_val = get_quanted_value_group<SourceType::kNumBits, false>(qb + qb_offset, idx2);
      }
      uint32_t shift_count = (kPaddedSrcBits * 4 % kTgtBits == 0) ? (kPaddedSrcBits * i % kTgtBits)
                                                                  : (kPaddedSrcBits * index % kTgtBits);
      if (shift_count) qb_val = qb_val << shift_count;
      res[i] = (qb_val & kKeepMaskRep) >> kHighPad;
    }
  }
}


template <class SourceType, class TargetType, bool kHasZeroPoint, bool kIsFpZeroPoint, uint32_t kNumWarpShapeNSplits = 1>
CUDA_INLINE void dequant_b3567(const uint32_t *qb, uint32_t *res, uint32_t j, uint32_t *zp_vals = nullptr) {
  static_assert(SourceType::kBits <= TargetType::kBits);
  constexpr uint32_t kPaddedNumBits = static_next_power_of_2(SourceType::kBits);
  constexpr bool kIsFpToFp = SourceType::kIsFloatingPointType && TargetType::kIsFloatingPointType;
  constexpr uint32_t reverse_pattern = TargetType::kBits / static_next_power_of_2(SourceType::kBits);
  uint32_t qb_val;

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 4; i++) {
    uint32_t zp_val = zp_vals[i];
    uint32_t index;

    if constexpr (kNumWarpShapeNSplits == 1 || SourceType::kBits == 6) {
      index = j * 4 + i;
      if (index * kPaddedNumBits % TargetType::kBits == 0) {
        const uint32_t idx1 = index / TargetType::kBits;
        const uint32_t idx2 = index * kPaddedNumBits / TargetType::kBits % kPaddedNumBits;
        const uint32_t qb_offset = idx1 * SourceType::kBits;
        qb_val = get_quanted_value_group<SourceType::kNumBits, !kIsFpToFp>(qb + qb_offset, idx2);
      }
    } else if (threadIdx.x / 32 % 2 == 0) {
      index = j * 4 + i;
      if (index * kPaddedNumBits % TargetType::kBits == 0) {
        const uint32_t idx1 = index / TargetType::kBits;
        const uint32_t idx2 = index * kPaddedNumBits / TargetType::kBits % kPaddedNumBits;
        const uint32_t qb_offset = idx1 * SourceType::kBits;
        qb_val = get_quanted_value_group<SourceType::kNumBits, !kIsFpToFp>(qb + qb_offset, idx2);
      }
    } else {
      index = (j + 2) * 4 + i;
      if (index * kPaddedNumBits % TargetType::kBits == 0) {
        const uint32_t idx1 = index / TargetType::kBits;
        const uint32_t idx2 = index * kPaddedNumBits / TargetType::kBits % kPaddedNumBits;
        const uint32_t qb_offset = idx1 * SourceType::kBits;
        qb_val = get_quanted_value_group<SourceType::kNumBits, !kIsFpToFp>(qb + qb_offset, idx2);
      }
    }

    uint32_t shift_count;
    if constexpr (kPaddedNumBits * 4 % TargetType::kBits == 0) {
      shift_count = kPaddedNumBits * i % TargetType::kBits;
    } else {
      shift_count = kPaddedNumBits * index % TargetType::kBits;
    }

    uint32_t qb_val2 = qb_val;
    if (kIsFpToFp && shift_count) qb_val2 = qb_val << shift_count;
    if (!kIsFpToFp && shift_count) qb_val2 = qb_val >> shift_count;

    if constexpr (!kIsFpToFp) {
      res[i] = dequant_single<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint>(qb_val2, zp_val);
    } else {
      uint32_t reversed_index = i / reverse_pattern * reverse_pattern + reverse_pattern - 1 - i % reverse_pattern;
      res[reversed_index] = dequant_single<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint>(qb_val2, zp_val);
    }
  }
}


template <class SourceType, class TargetType, bool kHasZeroPoint = false, bool kIsFpZeroPoint = false, uint32_t kNumWarpShapeNSplits = 1>
CUDA_INLINE void dequant(const uint32_t *qb, uint32_t *res, uint32_t j, uint32_t *zp_vals = nullptr) {
  if constexpr (SourceType::kBits == static_next_power_of_2(SourceType::kBits)) {
    dequant_b1248<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint>(qb, res, j, zp_vals);
  } else {
    dequant_b3567<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint, kNumWarpShapeNSplits>(qb, res, j, zp_vals);
  }
}
