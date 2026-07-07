#pragma once

#include <humming/scheduler.cuh>
#include <humming/utils/all.cuh>

#include <humming/arith/epilogue_arith.cuh>
#include <humming/arith/mainloop_arith.cuh>

#include <humming/epilogue/pipeline.cuh>
#include <humming/memory/g2s_pipeline.cuh>
#include <humming/memory/producer_dequant.cuh>
#include <humming/memory/s2r_pipeline.cuh>
#include <humming/mma/wgmma.cuh>
#include <humming/mma/wmma.cuh>

#include <humming/datatype/dequant.cuh>


template <bool kUseTma>
class KernelTensorParamType {
public:
  using Type = std::conditional_t<kUseTma, CUtensorMap const, void *const>;
};


template <
    class MmaOpClass,
    class ProblemShape, class BlockShape, class WarpShape, class PadShape,
    class ElementA, class ElementB, class ElementC, class ElementBS,
    class LayerConfig, class ComputeConfig, class TuningConfig>
__global__ __launch_bounds__(TuningConfig::kNumThreads, TuningConfig::kNumCtasPerSm) void humming(
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaA>::Type A,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaB>::Type B,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaC>::Type C,
    const uint32_t *AS,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBS>::Type BS,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBZP>::Type BZP,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBias>::Type Bias,
    const uint32_t *GS,
    const uint32_t *sorted_ids_ptr,
    const uint32_t *expert_ids_ptr,
    const uint32_t *num_tokens_padded_ptr,
    const uint32_t *expert_layout_ptr,
    CUtensorMap *tensor_map_buffer,
    int32_t *locks,
    uint32_t shape_m,
    uint32_t top_k,
    bool use_int64_expert_layout) {

  constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;
  constexpr uint32_t kNumStages = TuningConfig::kNumStages;

  using SharedStorage = SharedStorage<
      MmaOpClass, BlockShape, WarpShape, ElementA, ElementB, ElementBS,
      LayerConfig, ComputeConfig, TuningConfig>;

  // Producer may run into block i+1 without waiting for the consumer's
  // epilogue of block i. Requires the dedicated epilogue staging region
  // (SharedStorage::kFreeRunning) plus: whole blocks per slice (no stream-K)
  // and K_BLOCKS a multiple of the stage count, so a block's final stage lands
  // in slot kNumStages-1 and the initial loads of the next block (slots
  // 0..kNumStages-2) are all covered by already-consumed mbarrier arrivals.
  constexpr bool kFreeRunning =
      SharedStorage::kFreeRunning && !TuningConfig::kUseStreamK &&
      (ProblemShape::K / BlockShape::K) % kNumStages == 0;
  constexpr bool kDescMode =
      TuningConfig::kUseProducerDequant && LayerConfig::kInputScaleGroupSize > 0 &&
      LayerConfig::kInputScaleGroupSize <= BlockShape::K;
  using Scheduler = Scheduler<
      SharedStorage, ProblemShape, BlockShape,
      LayerConfig, ComputeConfig, TuningConfig>;
  using ProducerPipeline = ProducerPipeline<
      SharedStorage, ProblemShape, BlockShape, PadShape, ElementA, ElementB, ElementBS,
      LayerConfig, ComputeConfig, TuningConfig>;
  using ConsumerPipeline = ConsumerPipeline<SharedStorage, ElementA, LayerConfig, TuningConfig>;
  using MainloopArithmetic = MainloopArithmetic<
      MmaOpClass, BlockShape, WarpShape,
      ElementA, ElementB, ElementC, ElementBS, LayerConfig>;
  using EpilogueArithmetic = EpilogueArithmetic<
      MmaOpClass, BlockShape, WarpShape,
      ElementA, ElementB, ElementC, ElementBS,
      LayerConfig, TuningConfig>;
  using WMMA = WMMA<MmaOpClass, SharedStorage, MainloopArithmetic, WarpShape, ElementA, ElementB, LayerConfig>;
  using WGMMA = WGMMA<MmaOpClass, SharedStorage, MainloopArithmetic, BlockShape, WarpShape, ElementA, ElementB, LayerConfig>;
  using MMA = std::conditional_t<MmaOpClass::kMmaType == MmaType::WGMMA, WGMMA, WMMA>;
  using Epilogue = EpiloguePipeline<
      MmaOpClass, SharedStorage, EpilogueArithmetic, ProblemShape, BlockShape, WarpShape, PadShape,
      ElementA, ElementC, LayerConfig, ComputeConfig, TuningConfig>;
  using S2RMemoryPipeline = S2RMemoryPipeline<
      SharedStorage, MMA, Epilogue, BlockShape, WarpShape, ElementA, ElementB, ElementBS,
      LayerConfig, TuningConfig>;

  extern __shared__ int4 shared_memory[];
  auto &smem = *reinterpret_cast<SharedStorage *>(shared_memory);

  auto pa = [&]() {if constexpr (TuningConfig::kUseTmaA) return &A; else return A; };
  auto pb = [&]() {if constexpr (TuningConfig::kUseTmaB) return &B; else return B; };
  auto pc = [&]() {if constexpr (TuningConfig::kUseTmaC) return &C; else return C; };
  auto pas = [&]() { return AS; };
  auto pbs = [&]() {if constexpr (TuningConfig::kUseTmaBS) return &BS; else return BS; };
  auto pbzp = [&]() {if constexpr (TuningConfig::kUseTmaBZP) return &BZP; else return BZP; };
  auto pbias = [&]() {if constexpr (TuningConfig::kUseTmaBias) return &Bias; else return Bias; };
  auto scheduler = Scheduler(smem, pc(), tensor_map_buffer, shape_m, top_k, sorted_ids_ptr, expert_ids_ptr, num_tokens_padded_ptr, expert_layout_ptr, use_int64_expert_layout);
  if (threadIdx.x >= TuningConfig::kNumThreads - TuningConfig::kNumLoadThreads) {
    // Producer-dequant: at 256 threads every thread already has the full
    // budget; with more threads rebalance so the math warpgroups get 184,
    // the loader warpgroup 40, and the dequant warpgroup the remainder.
    if constexpr (TuningConfig::kUseProducerDequant && TuningConfig::kNumDequantThreads > 0) {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(40));
    } else if constexpr (TuningConfig::kUseProducerDequant && TuningConfig::kNumThreads > 256) {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(96));
    } else if constexpr (TuningConfig::kUseProducerDequant) {
    } else if constexpr (TuningConfig::kNumMathThreads > 256) {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(40));
    } else if constexpr (TuningConfig::kNumCtasPerSm == 1 && ElementA::kBits != 16) {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(40));
    } else {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(24));
    }

    auto producer = ProducerPipeline(smem, pa(), pb(), pas(), pbs(), pbzp(), pbias(), shape_m);
    using ProducerDequant = ProducerDequantB<
        MmaOpClass, SharedStorage, BlockShape, WarpShape,
        ElementA, ElementB, ElementBS, LayerConfig, TuningConfig>;
    [[maybe_unused]] auto dq = [&]() {
      if constexpr (TuningConfig::kUseProducerDequant) return ProducerDequant(smem);
      else return 0;
    }();
    producer.init_mbarrier();
    __syncthreads();
    while (scheduler.get_next_block()) {
      uint32_t &slice_iters = scheduler.slice_iters;

      // Without a dedicated epilogue staging region (kFreeRunning), the
      // consumer's epilogue for the previous block reads smem the stage
      // buffers alias, so staging and stage-0 loads must wait for it. With
      // kFreeRunning the stage-slot mbarrier ring alone is sufficient: the
      // producer's next-block slot writes are gated by the consumer's
      // per-stage arrivals, the epilogue staging is separate smem, and the
      // row indices are double-buffered by block parity.
      if constexpr (!kFreeRunning) {
        producer.wait_math_epilogue();
      }
      if constexpr (ComputeConfig::kGemmType == GemmType::INDEXED) {
        scheduler.fetch_moe_index_block();
      }
      producer.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.k_block_id, scheduler.current_shape_m, scheduler.m_offset);
      producer.load_stage<true, true>(0);
      PRAGMA_UNROLL
      for (uint32_t stage_id = 1; stage_id < kNumStages - 1; stage_id++) {
        producer.load_stage(stage_id, stage_id < slice_iters);
      };

      bool first_block_iter = true;
      while (slice_iters) {
        PRAGMA_UNROLL
        for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
          if (slice_iters == 1) producer.load_channel();
          // Dequant BEFORE the math-mbar wait: it can run as soon as the
          // previous consumer arrive fired, overlapping the wait window.
          // With a dedicated dequant warpgroup this loop only issues loads.
          if constexpr (TuningConfig::kUseProducerDequant && TuningConfig::kNumDequantThreads == 0) {
            dq.process_stage(stage_id, first_block_iter && stage_id == 0);
          }
          producer.wait_stage(stage_id);
          producer.load_stage(stage_id + kNumStages - 1, slice_iters >= kNumStages);
          slice_iters--;
          if (!slice_iters) break;
        }
        first_block_iter = false;
      }
    }
  } else if (TuningConfig::kNumDequantThreads > 0 && threadIdx.x >= TuningConfig::kNumMathThreads) {
    // Dedicated dequant warpgroup: consumes quantized stages the loader
    // group cp.async'd, produces canonical FP8 tiles for the consumer's
    // wgmma descriptors. Runs its own scheduler walk; synchronizes with the
    // loader via load_mbar (data ready) and with the consumer via math_mbar
    // (bf8 slot free; the consumer arrives post-fold in desc mode) and
    // dq_mbar (tile ready).
    if constexpr (TuningConfig::kNumDequantThreads > 0) {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(88));

      using ProducerDequant = ProducerDequantB<
          MmaOpClass, SharedStorage, BlockShape, WarpShape,
          ElementA, ElementB, ElementBS, LayerConfig, TuningConfig>;
      auto dq = ProducerDequant(smem);
      __syncthreads();

      uint32_t dq_iteration = 0;
      while (scheduler.get_next_block()) {
        uint32_t &slice_iters = scheduler.slice_iters;
        bool first_block_iter = true;
        while (slice_iters) {
          PRAGMA_UNROLL
          for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
            // bf8 slot reuse licence: skip while the ring is first filling.
            if (dq_iteration >= kNumStages) dq.wait_math_licence(stage_id);
            dq.process_stage(stage_id, first_block_iter && stage_id == 0);
            dq_iteration++;
            slice_iters--;
            if (!slice_iters) break;
          }
          first_block_iter = false;
        }
      }
    }
  } else {
    if constexpr (TuningConfig::kNumMathThreads > 256) {
      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(96));
    } else if constexpr (TuningConfig::kUseProducerDequant && TuningConfig::kNumThreads > 256) {
      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(184));
    } else if constexpr (!TuningConfig::kUseProducerDequant) {
      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(232));
    }

    auto mainloop_arith = MainloopArithmetic();
    auto epilogue_arith = EpilogueArithmetic();
    auto mma = MMA(smem, mainloop_arith);
    auto epilogue = Epilogue(smem, pc(), tensor_map_buffer, epilogue_arith, GS, locks, shape_m, top_k);
    auto consumer = ConsumerPipeline(smem);
    auto s2r_pipe = S2RMemoryPipeline(smem, mma, epilogue);

    consumer.init_mbarrier();
    __syncthreads();
    if constexpr (!kFreeRunning) consumer.arrive(kNumStages);

    while (scheduler.get_next_block()) {
      mma.zero_accum();

      uint32_t &slice_iters = scheduler.slice_iters;
      epilogue.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.current_shape_m, scheduler.m_offset, scheduler.row_index_parity);
      epilogue.set_streamk_state(scheduler.slice_count, scheduler.slice_id, scheduler.locks_offset);

      consumer.wait_stage<true>(kNumStages);
      s2r_pipe.load_stage_iter<true>(0, 0);
      mma.transform_b(0);

      while (slice_iters) {
        PRAGMA_UNROLL
        for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
          constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
          constexpr uint32_t warp_k_iters = WarpShape::K / kPartMmaShapeK;

          // In desc mode the bf8 tile is first touched by this stage's first
          // wgmma, so the dequant handshake waits here (a full producer
          // stage of slack) instead of inside the previous stage's k-2 slot.
          consumer.wait_dq(stage_id);

          PRAGMA_UNROLL
          for (uint32_t warp_k_iter_id = 0; warp_k_iter_id < warp_k_iters; warp_k_iter_id++) {
            if constexpr (TuningConfig::kUseProducerDequant || std::is_same<ElementA, ElementB>::value) {
              // s2r writes regs_b directly here, so it must come AFTER run():
              // run's wait<1> drains the group that was still reading the
              // regs_b buffer the next s2r refills.
              mma.run(stage_id, warp_k_iter_id);
              if (warp_k_iter_id == warp_k_iters - 2) {
                // Desc mode arrives post-fold instead (below): the arrive is
                // the dequant group's licence that all wgmma reads of this
                // stage's bf8 tile have completed.
                if constexpr (!kDescMode) consumer.arrive(stage_id);
                if (slice_iters > 1) {
                  consumer.wait_stage((stage_id + 1) % kNumStages);
                }
              }
              s2r_pipe.load_stage_iter(stage_id, warp_k_iter_id + 1);
            } else {
              s2r_pipe.load_stage_iter(stage_id, warp_k_iter_id + 1);
              mma.run(stage_id, warp_k_iter_id);
              if (warp_k_iter_id == warp_k_iters - 2) {
                consumer.arrive(stage_id);
                if (slice_iters > 1) {
                  consumer.wait_stage((stage_id + 1) % kNumStages);
                  consumer.wait_dq((stage_id + 1) % kNumStages);
                }
              }

              mma.transform_b((warp_k_iter_id + 1) % 2);
            }
          }

          if constexpr (kDescMode) consumer.arrive(stage_id);

          slice_iters--;
          if (!slice_iters) break;
        };
      };

      mma.wait_all();
      consumer.wait_channel();
      s2r_pipe.load_channel(scheduler.slice_id);
      epilogue.call(mma.final_regs_c_as_ptr());
      if constexpr (TuningConfig::kUseTmaC) tma_wait_store_group<0, true>();
      if constexpr (!kFreeRunning) consumer.arrive(kNumStages);
    }
  }

  __syncthreads();
  if constexpr (TuningConfig::kMultiCastSizeA > 0 || TuningConfig::kMultiCastSizeB > 0) {
    asm volatile("barrier.cluster.arrive;\n");
    asm volatile("barrier.cluster.wait;\n");
  }
};
