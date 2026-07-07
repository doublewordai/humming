#pragma once

#include <humming/datatype/dequant_fused.cuh>
#include <humming/memory/s2r_loader/loader_b.cuh>
#include <humming/memory/s2r_loader/loader_bs.cuh>
#include <humming/utils/all.cuh>

// Producer-side dequantization for the warp-specialized indexed path.
//
// The producer warp group replays the consumer's s2r fragment loads (the
// quantized-B smem layout is thread-strided, and producer warps alias math
// warps modulo N_WARPS), runs the same fused MXFP4+E8M0 -> FP8 dequant the
// consumer's transform_b would have run, and stores the resulting fragments
// to a dedicated FP8 stage buffer (smem.bf8) laid out exactly as
// S2RMemoryLoaderB<..., ElementB = ElementA> expects, so the consumer loads
// ready-to-issue wgmma A fragments with two LDS.128 per iteration and issues
// no dequant instructions at all.
template <
    class MmaOpClass, class SharedStorage,
    class BlockShape, class WarpShape,
    class ElementA, class ElementB, class ElementBS,
    class LayerConfig, class TuningConfig>
class ProducerDequantB {
private:
  static_assert(LayerConfig::kUseFusedE8m0Scale);
  static_assert(BlockShape::K == WarpShape::K, "producer dequant assumes K_WARPS == 1");

  static constexpr uint32_t kNumStages = TuningConfig::kNumStages;
  static constexpr uint32_t kNumLoadThreads = TuningConfig::kNumLoadThreads;
  static constexpr uint32_t kNumMathThreads = TuningConfig::kNumMathThreads;
  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kWarpItersK = WarpShape::K / kPartMmaShapeK;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  // Each producer warp covers the fragments of the math warps it aliases
  // (warp_id % N_WARPS); with kNumLoadThreads == kNumMathThreads the aliasing
  // is 1:1.
  static_assert(kNumLoadThreads == kNumMathThreads,
                "producer dequant requires one producer warp per math warp");

  // fused_dequant_for_mxfp4<.., kCount = WarpShape::N/16> consumes kCount*2
  // quant words and emits two FP8 words per quant word.
  static constexpr uint32_t kNumQBWords = WarpShape::N / 16 * 2;
  static constexpr uint32_t kNumOutWords = kNumQBWords * 2;

  // Mirror of S2RMemoryLoaderB<..., ElementB=ElementA> addressing (the
  // consumer-side reader of bf8).
  static constexpr uint32_t kSmemStrideBF8 = BlockShape::N * kPartMmaShapeK * ElementA::kBits / 32 / 4;
  static constexpr uint32_t kOutIntsPerThread = ElementA::kBits;
  static constexpr uint32_t kOutInt4sPerThread = kOutIntsPerThread / 4;
  static constexpr uint32_t kWarpWeightBlocks = MAX(WarpShape::N / (ElementA::kBits * 4), 1);

  using LoaderQB = S2RMemoryLoaderB<BlockShape, WarpShape, ElementA, ElementB, TuningConfig>;
  using LoaderBS = S2RMemoryLoaderBS<MmaOpClass, BlockShape, WarpShape, ElementA, ElementBS, LayerConfig, TuningConfig>;

public:
  SharedStorage &smem;
  LoaderQB loader_qb;
  LoaderBS loader_bs;
  // Statically indexed (stage_id is an unroll constant; is_first is a
  // separate scalar) — a runtime-indexed phase array demotes to local memory.
  uint32_t slot_phases[kNumStages] = {0};
  uint32_t first_phase = 0;
  alignas(16) uint32_t regs_qb[kNumQBWords];
  alignas(16) uint32_t regs_bs[2];
  alignas(16) uint32_t regs_out[kNumOutWords];

  CUDA_INLINE
  ProducerDequantB(SharedStorage &smem)
      : smem(smem) {
  }

  // Dequantize one k-block worth of B fragments for `stage_id`.
  // `is_first_stage_commit` selects the block-initial mbarrier index used by
  // load_stage<kIsFirst=true> for stage 0.
  CUDA_INLINE
  void process_stage(uint32_t stage_id, bool is_first_stage_commit) {
    if (is_first_stage_commit) {
      mbarrier_wait(&smem.load_mbar[kNumStages], first_phase);
      first_phase ^= 1;
    } else {
      mbarrier_wait(&smem.load_mbar[stage_id], slot_phases[stage_id]);
      slot_phases[stage_id] ^= 1;
    }

    PRAGMA_UNROLL
    for (uint32_t iter_id = 0; iter_id < kWarpItersK; iter_id++) {
      loader_qb.load(smem.b[stage_id], regs_qb, iter_id);
      loader_bs.load(smem.bs[stage_id], regs_bs, iter_id);
      fused_dequant_for_mxfp4<ElementA, WarpShape::N / 16, true>(regs_qb, regs_out, regs_bs);
      store_fragments(stage_id, iter_id);
    }

    if (threadIdx.x % 32 == 0) mbarrier_arrive(&smem.dq_mbar[stage_id]);
    __syncwarp();
  }

  CUDA_INLINE
  void store_fragments(uint32_t stage_id, uint32_t iter_id) {
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t n_warp_id = warp_id % N_WARPS;
    uint32_t lane_id = threadIdx.x % 32;

    uint32_t idx = kWarpWeightBlocks * 32 * n_warp_id + lane_id;
    int4 *smem_ptr = smem.bf8[stage_id] + kSmemStrideBF8 * iter_id;
    int4 *out_int4 = reinterpret_cast<int4 *>(regs_out);

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kWarpWeightBlocks; i++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kOutInt4sPerThread / kWarpWeightBlocks; j++) {
        constexpr uint32_t kInt4sPerBlock = kOutInt4sPerThread / kWarpWeightBlocks;
        smem_ptr[(idx + 32 * i) * kInt4sPerBlock + j] = out_int4[i * kInt4sPerBlock + j];
      }
    }
  }
};
