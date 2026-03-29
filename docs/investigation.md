# Investigation: Achieving CPU-GPU Bitwise Parity in llama.cpp

## Summary

Over 3 sessions, we identified and fixed 18 sources of floating-point divergence between CUDA GPU (RTX 3090, SM86) and x86 CPU inference in llama.cpp. The result: every intermediate tensor (2462 total across 36 transformer layers) is bitwise identical, and token output matches at temp 0.7 through full autoregressive decode.

Model tested: `qwen2.5-3b-instruct-q4_k_m.gguf` (Q4_K_M quantization, ~2GB).

## Methodology

**Diagnostic tool:** `det-diag` — hooks into llama.cpp's eval callback to compute FNV-1a hash of every tensor during inference. Run on GPU and CPU, diff the hashes to find the first diverging operation.

**Approach:** Fix divergences in layer order. Each fix extends the matching prefix until the next divergence appears. Verified with the diagnostic after each fix.

**PTX/SASS analysis:** Used `nvcc --ptx` and `cuobjdump --dump-sass` to determine exact GPU floating-point instruction sequences, then matched on CPU.

## Timeline of fixes

### Session 1: RMSNorm + MatMul foundations

**Phase 0 — Remove `-use_fast_math`** from CUDA build when `GGML_DETERMINISTIC=ON`. Without this, CUDA uses approximate transcendentals (~5 ULP error). File: `ggml-cuda/CMakeLists.txt`.

**Phase 1 — RMSNorm butterfly reduction.** The GPU reduces partial sums using `__shfl_xor_sync` (simultaneous read-then-write across warp). The CPU used a sequential loop which reads stale values. Fixed with a double-buffered `buf[2][32]` alternating read/write per butterfly stage. File: `ggml-cpu/ops.cpp`.

**Phase 2 — Q4_K dot product with explicit FMA.** NVCC auto-fuses `sumf_d += d8[i] * (dot1 * sc[i])` into `fma.rn.f32`. The CPU without `-mfma` does separate multiply+add, differing by 1 ULP. Fixed with explicit `fmaf()` calls matching NVCC's fusion pattern. Confirmed by PTX analysis. File: `ggml-cpu/quants.c`.

**Phase 3 — Attention FP32 V accumulation.** GPU accumulates V in FP32; CPU was using FP16. Fixed for deterministic mode.

**Phase 4 — Portable `expf_det`.** CUDA's device `expf()` compiles to `ex2.approx.ftz.f32` — a silicon lookup table in the SM, unreproducible in software. 2982/10000 values differ from glibc. Solution: portable Cody-Waite + Horner polynomial implementation, 0/100000 mismatches. Used for SiLU on both sides. Files: `ggml-deterministic.h`, `unary.cu`, `vec.cpp`.

**Phase 5 — RoPE `powf`.** GPU uses `powf(theta_scale, i0/2)` per-dimension instead of iterative `theta *= theta_scale`. Iterative version accumulates FP32 rounding error. Fixed CPU to use `powf` per-dimension.

**Phase 6 — Q8_1 quantization.** CPU used `roundf(xi * (1/d))` (reciprocal multiply); GPU uses `roundf(xi / d)` (direct division). These differ due to FP32 rounding of `1/d`. Fixed for deterministic mode. File: `ggml-quants.c`.

**Phase 7 — Softmax float accumulation.** Deterministic path uses FP32 instead of double for softmax accumulation, matching GPU.

**Result after Session 1:** ~94% tensor match, 13 clean transformer layers.

### Session 2: (continuation, context compacted)

Continued debugging the layer 13 `ffn_gate` matmul divergence. Established per-tensor diagnostic infrastructure.

### Session 3: RoPE, attention, and the x86 quantization breakthrough

**Phase 8 — Portable `powf_int_det`.** CUDA device `powf()` differs from glibc by ±1 ULP (4/64 mismatches for RoPE theta values). Binary exponentiation using only IEEE multiply — 0 mismatches. Files: `ggml-deterministic.h`, `rope.cu`, `ops.cpp`.

**Phase 9 — Portable `sincosf_det`.** CUDA device `sinf`/`cosf` differ from glibc by ±1 ULP (6/64 combined). Portable Cody-Waite + minimax polynomial — 0 mismatches.

**Phase 10 — `-mfma` CPU build.** Without hardware FMA support, GCC generates separate multiply+add for `a*b+c`, which differs from NVCC's auto-fused FMA by 1 ULP. Adding `-mfma` lets GCC emit `vfmadd` instructions matching the GPU. Essential for RoPE rotation parity.

**Phase 11 — Tile-based CPU attention.** GPU flash attention processes K-V positions in tiles of D=128 with parallel softmax. CPU used sequential online softmax. Rewrote CPU to compute all Q·K values first, find max, compute softmax, then accumulate V — matching GPU tile structure.

**Phase 12 — FP32 Q·K dot product.** GPU uses FP32 with pre-scaled Q and warp butterfly reduction. CPU was converting Q to F16 and using F16 precision. Rewrote with FP32 dot product + 32-lane butterfly matching GPU warp reduction.

**Phase 13 — Direct division `VKQ/S`.** GPU computes `dst_val /= kqsum` (one division). CPU used `S_inv = 1.0f/S; VKQ *= S_inv` (reciprocal + multiply = two roundings). Changed to direct `VKQ[d] /= S`.

**Phase 14 — `expf_det` in attention softmax.** Both GPU and CPU flash attention used platform `expf()` for softmax weights. Replaced with `expf_det` on both sides.

**Phase 15 — `__frsqrt_rn` in GPU RMSNorm.** CUDA's `rsqrtf()` uses `rsqrt.approx.f32` — a hardware approximation with ≤2 ULP error, even without fast-math. Replaced with `__frsqrt_rn()` (correctly rounded) on GPU. CPU uses `(float)(1.0/sqrt((double)x))` which matches.

**Phase 16 — Q6_K explicit outer FMA.** GPU fuses `acc += d6 * partial` into `fma(d6, partial, acc)`. Made explicit with `fmaf()` on CPU. Verified by SASS analysis of the Q6_K mmvq kernel.

**Phase 17 — x86 Q8_1 quantization bypass.** **The hidden blocker.** The x86 SIMD version of `quantize_row_q8_1` in `arch/x86/quants.c` uses `roundf(x * 127/max)` (reciprocal multiply) while the GPU uses `roundf(x / d)` where `d = max/127`. Our fix in `quantize_row_q8_1_ref` was correct but was never called — the x86 override took priority. And `-mfma` enables `__AVX__`, activating the x86 path. Fixed by adding a deterministic early-return to the ref implementation.

**This single fix jumped us from 1 clean layer to 35 clean layers.**

**Phase 18 — Per-column mmvq.** The GPU mmvq kernel is templated on `ncols_dst`. Different template instantiations (ncols=4 vs ncols=7) generate different ptxas SASS, even for mathematically identical per-column operations. Fixed by processing one column at a time when deterministic. Also extended mmvq to handle batch sizes > `MMVQ_MAX_BATCH_SIZE` by looping over columns.

**Result: 2462/2462 tensors match. 100% parity.**

## Key technical insights

### 1. Every CUDA transcendental is wrong (for reproducibility)

| Function | CUDA implementation | Mismatch rate vs glibc |
|----------|-------------------|----------------------|
| `expf()` | `ex2.approx.ftz.f32` hardware LUT | 30% |
| `sinf()` | Software polynomial (differs from glibc) | 6% |
| `cosf()` | Software polynomial (differs from glibc) | 6% |
| `powf()` | Software (differs from glibc) | 6% |
| `rsqrtf()` | `rsqrt.approx.f32` hardware LUT | ~50% |

Solution: portable implementations using only FMA + basic IEEE ops, shared between GPU and CPU.

### 2. PTX is not SASS

NVCC generates PTX (intermediate representation), but `ptxas` (the GPU JIT assembler) further optimizes into SASS (actual machine code). PTX showing `mul.f32 + sub.f32` might become `FFMA` in SASS. Must verify with `cuobjdump --dump-sass`, not just `--ptx`.

For our case: isolated kernels showed PTX matching natural arithmetic. The divergence was from x86 quantization, not JIT restructuring.

### 3. Architecture-specific code silently overrides

The x86 SIMD `quantize_row_q8_1` in `arch/x86/quants.c` overrides the generic version in `ggml-quants.c`. Adding `-mfma` enables `__AVX__`, activating this override. Our deterministic fix in the generic version was never called. **Always check `arch/*/quants.c`.**

### 4. Attention tile structure matters

GPU flash attention processes K-V positions in tiles of D (head dimension, typically 128), computing softmax per-tile with rescaling between tiles. CPU's sequential online softmax produces different floating-point results because additions happen in a different order. For bitwise parity, the CPU must match the tile structure exactly.

### 5. Batch size changes everything

The GPU mmvq kernel is a C++ template parameterized on `ncols_dst` (batch size). Each instantiation (ncols=1, ncols=4, ncols=7, etc.) is a separate compiled kernel. `ptxas` may optimize each differently. For deterministic output, process one column at a time regardless of batch size.

## GPU environment specification

The GPU reference output is determined by:
- **llama.cpp commit:** `d092e2682cc1db9f33b158b4378b448897b3096c`
- **This patch set:** all 18 phases
- **CUDA toolkit:** 12.8 (`/usr/local/cuda-12.8/bin/nvcc`)
- **GPU architecture:** SM86 (RTX 3090)
- **Build flags:** `-DGGML_DETERMINISTIC=ON`, no `-use_fast_math`

Changing any of these may change the GPU output. The CPU verifier is independent of GPU hardware.

## Current limitations

- Only Q4_K_M quantization verified (Q4_K + Q6_K weight types)
- IQ4_XS, Q5_K, Q8_0 need additional deterministic dot products
- MoE routing layer (F32 matmul) needs work — the 30B MoE model matches through attention but diverges at the router
- Tested on SM86 only — other GPU architectures may need re-verification
- This is a fork, not upstream — rebasing to latest llama.cpp needed for newer model architectures
