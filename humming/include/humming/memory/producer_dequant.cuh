#pragma once

#include <humming/datatype/dequant_fused.cuh>
#include <humming/memory/s2r_loader/loader_b.cuh>
#include <humming/memory/s2r_loader/loader_bs.cuh>
#include <humming/utils/all.cuh>

// Producer-side dequantization for the warp-specialized indexed path.
//
// The producer warp group replays the consumer's s2r fragment loads (the
// quantized-B smem layout is thread-strided; producer warps cover math warps
// via an explicit warp alias), runs the same fused MXFP4+E8M0 -> FP8 dequant
// the consumer's transform_b would have run, and stores the result to a
// dedicated FP8 stage buffer (smem.bf8) in CANONICAL wgmma-A layout so the
// consumer feeds both wgmma operands by shared-memory descriptor and issues
// no dequant or B-fragment loads at all.
//
// Canonical layout (empirically verified against the repack kernel,
// kernel-dev/probe_pack.py, 0/65536 mismatches): the packed word at uint4
// index (32*wp + lane), word wi of a 32-K slab holds
//   n = 32*wp + 8*wi + lane/4,  k = 32*slab + 16*half + 4*(lane%4) + byte
// After fused dequant (+ the wgmma pair swap), output word o (0..7) is the
// 4 consecutive FP8 values at
//   n(o) = 32*wp + 16*(o/4) + 8*(o%2) + lane/4
//   k(o) = 32*iter + 16*((o%4)/2) + 4*(lane%4) + 0..3
// The desc-consumed m64 tile (warpgroup g = wp/4, j = o/4) wants value n at
// tile row d = 16*(wp%4) + lane/4 + 8*(o%2), matching the D-fragment rows the
// epilogue already expects, so the output mapping is unchanged.
template <
    class MmaOpClass, class SharedStorage,
    class BlockShape, class WarpShape,
    class ElementA, class ElementB, class ElementBS,
    class LayerConfig, class TuningConfig>
class ProducerDequantB {
private:
  static_assert(LayerConfig::kUseFusedE8m0Scale);
  static_assert(BlockShape::K == WarpShape::K, "producer dequant assumes K_WARPS == 1");
  static_assert(BlockShape::K * ElementA::kBits == 1024,
                "canonical bf8 rows are one 128B swizzle atom");

  static constexpr uint32_t kNumStages = TuningConfig::kNumStages;
  static constexpr uint32_t kNumLoadThreads = TuningConfig::kNumLoadThreads;
  static constexpr uint32_t kNumMathThreads = TuningConfig::kNumMathThreads;
  static constexpr uint32_t kNumDequantThreads =
      TuningConfig::kNumDequantThreads ? TuningConfig::kNumDequantThreads : kNumLoadThreads;
  static constexpr uint32_t kNumProducerWarps = kNumDequantThreads / 32;
  static constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
  static constexpr uint32_t kNumReps = kNumMathWarps / kNumProducerWarps;
  static_assert(kNumMathWarps % kNumProducerWarps == 0);

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kWarpItersK = WarpShape::K / kPartMmaShapeK;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;

  // fused_dequant_for_mxfp4<.., kCount = WarpShape::N/16> consumes kCount*2
  // quant words and emits two FP8 words per quant word.
  static constexpr uint32_t kNumQBWords = WarpShape::N / 16 * 2;
  static constexpr uint32_t kNumOutWords = kNumQBWords * 2;

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
  uint32_t math_phases[kNumStages] = {0};
  // Double-buffered so the next step's quant/scale loads issue before the
  // current step's dequant consumes its registers: a single producer warp per
  // scheduler has nothing else to hide the LDS latency behind (measured
  // long_scoreboard 4.52 cycles/issued-instruction without the prefetch).
  alignas(16) uint32_t regs_qb[2][kNumQBWords];
  alignas(16) uint32_t regs_bs[2][2];
  alignas(16) uint32_t regs_out[kNumOutWords];

  CUDA_INLINE
  ProducerDequantB(SharedStorage &smem)
      : smem(smem) {
  }

  // bf8 overwrite licence for the dedicated dequant warpgroup: the consumer
  // arrives math_mbar[stage] post-fold in desc mode, so an observed arrival
  // means every wgmma that read bf8[stage] has completed.
  CUDA_INLINE
  void wait_math_licence(uint32_t stage_id) {
    mbarrier_wait(&smem.math_mbar[stage_id], math_phases[stage_id]);
    math_phases[stage_id] ^= 1;
  }

  // Dequantize one k-block worth of B for `stage_id` into canonical FP8.
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

    uint32_t producer_warp = (threadIdx.x - kNumMathThreads) / 32;
    constexpr uint32_t kNumSteps = kNumReps * kWarpItersK;
    static_assert(kNumSteps >= 2);

    loader_qb.load(smem.b[stage_id], regs_qb[0], 0, producer_warp);
    loader_bs.load(smem.bs[stage_id], regs_bs[0], 0, producer_warp);

    PRAGMA_UNROLL
    for (uint32_t step = 0; step < kNumSteps; step++) {
      uint32_t rep = step / kWarpItersK;
      uint32_t iter_id = step % kWarpItersK;
      uint32_t wp = producer_warp + kNumProducerWarps * rep;

      if (step + 1 < kNumSteps) {
        uint32_t nrep = (step + 1) / kWarpItersK;
        uint32_t niter = (step + 1) % kWarpItersK;
        uint32_t nwp = producer_warp + kNumProducerWarps * nrep;
        loader_qb.load(smem.b[stage_id], regs_qb[(step + 1) % 2], niter, nwp);
        loader_bs.load(smem.bs[stage_id], regs_bs[(step + 1) % 2], niter, nwp);
      }

      fused_dequant_for_mxfp4<ElementA, WarpShape::N / 16, true>(regs_qb[step % 2], regs_out, regs_bs[step % 2]);
      store_canonical(stage_id, iter_id, wp);
    }

    if (threadIdx.x % 32 == 0) mbarrier_arrive(&smem.dq_mbar[stage_id]);
    __syncwarp();
  }

  CUDA_INLINE
  void store_canonical(uint32_t stage_id, uint32_t iter_id, uint32_t wp) {
    uint32_t lane = threadIdx.x % 32;
    uint8_t *base = reinterpret_cast<uint8_t *>(smem.bf8[stage_id]);
    // SW128 phase depends on the absolute row address; tiles are 64 rows so
    // only the buffer base and the in-tile row matter mod 8.
    uint32_t phase_base = cast_smem_ptr_to_uint(smem.bf8[stage_id]) / 128 % 8;

    PRAGMA_UNROLL
    for (uint32_t o = 0; o < kNumOutWords; o++) {
      uint32_t tile = (wp / 4) * 2 + (o / 4);
      uint32_t d = 16 * (wp % 4) + lane / 4 + 8 * (o % 2);
      uint32_t col = 32 * iter_id + 16 * ((o % 4) / 2) + 4 * (lane % 4);
      uint32_t chunk = col / 16;
      uint32_t swizzled = (chunk ^ ((d + phase_base) % 8)) * 16 + col % 16;
      uint32_t byte_off = tile * (64 * 128) + d * 128 + swizzled;
      *reinterpret_cast<uint32_t *>(base + byte_off) = regs_out[o];
    }
  }
};
