# Doubleword Humming integration

The September GH200 qualification uses release `v0.1.12`, commit
`636ba85648c30ae2c2bfb335c9399593a67ecc1d`, with two small patch branches:

- `upstream-pr/accumulator-access-safety`: align register accumulator storage
  and use scalar FP32 accesses for the shared-memory reduction.
- `upstream-pr/moe-valid-rows`: quantize only live routed rows, optionally fuse
  clamped SwiGLU, and build indexed metadata from device-resident expert counts.

The installation remains a normal `humming-kernels` wheel. vLLM imports the
installed package and supplies its supported geometry/tile policy. Humming
contains no vLLM imports, source-path injection, or model-specific dispatcher.

## Reconciliation of the earlier fork

The former integration head was `a07f4b35089eb84213fe1e258a727861d880aed1`,
based on `79c1cef7d65665d0446137210120b54433a14e87`. Preserve both as
`archive/main-pre-v0.1.12-20260915` and `archive/base-pre-v0.1.12-20260915` when
publishing the new integration. Historical patch PRs belong to that archived
base; their implementations are not silently reapplied to the refactored
release.

| Earlier patch | Treatment in the new stack |
| --- | --- |
| `6290dc8`, input group-size plumbing | Release `humming.forward.may_quant_input` already forwards `input_scale_group_size` and scale dtype. |
| `f4a3694`, batched group-scale fold | Replaced by the release's WGMMA/mainloop implementation, including group-scale accumulation and batched issue/wait. Retaining the old header diff would revert unrelated release fixes. The new indexed W4A8 tests exercise varying per-group scales. |
| `7772e8f`, SM90 M-tile cap | The qualified engine supplies explicit M8/M64 configurations for the tested model geometry. Generic release heuristics remain unchanged. |
| `e0dc475` and `1cbe3d1`, fused activation/quantization | Replaced by the valid-row and sorted-row producers, including explicit BF16 rounding and masked padding. The earlier standalone symbol is not retained. |
| `6fac1fe`, quantization group batching | The selected MoE path uses the new live-row producers. The release's generic quantizer is retained. |
| `2094c2e`, vLLM group-quant routing | vLLM supplies dispatched FP8 values and FP32 scales through Humming's normal input-scale interface. Humming no longer needs a vLLM-specific quantizer route. |
| `8527fd6`, `HUMMING_DEBUG_CONFIG` | Retired diagnostic flag; archived with the old branch. |
| `8aaaf20`, `HUMMING_FORCE_HEURISTICS` | Retired diagnostic override; callers can supply explicit tuning configurations. |
| `891b132`, `HUMMING_INDEXED_MAX_BLOCK_M` | Replaced in the qualified path by the explicit engine tile policy. The old global override is archived. |

This is a deliberate release alignment, not a claim that every historical API
or environment variable is unchanged. The qualification package was already
based on `v0.1.12`; the reconciliation makes the published source match that
package.

## Validation

A clean wheel installation from the integration source passed 22 GPU tests:
valid/sorted quantization, changing device-count metadata under CUDA graph
replay, non-power-of-two masked capacities, and indexed MXFP4/FP8 accumulation
at M8/M16/M32 with stream-K both enabled and disabled. The engine integration
has a separate installed-wheel and serving validation gate.
