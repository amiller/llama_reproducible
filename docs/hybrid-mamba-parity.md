# Hybrid Mamba Parity: Extending Deterministic Inference to Linear Attention

## Summary

Sessions 4-6 extended CPU-GPU bitwise parity from standard transformers (Q4_K_M) to hybrid Mamba/linear-attention architectures (IQ4_XS), culminating in full parity for Qwen3.5 models from 4B to 27B parameters. This required fixing 8 additional sources of divergence beyond the 18 documented in `investigation.md`, for a total of 26.

The Qwen3.5 architecture is 75% linear attention (gated delta net + 1D convolution) and 25% full attention — a fundamentally different computation graph that exercises code paths untouched by standard transformers.

## Why hybrid models are harder

Standard transformers have a relatively uniform computation graph: matmul → norm → attention → matmul → norm → FFN, repeated N times. Every operation is either a quantized dot product, a reduction (softmax/norm), or a transcendental (SiLU, RoPE). Once you fix those building blocks, all transformer models work.

Hybrid Mamba models introduce:
- **Gated delta net (GDN)**: recurrent state updates with `exp(g) * S + k * delta`, requiring portable expf in a fused CUDA kernel
- **1D convolution (SSM conv)**: FMA contraction in a 3-tap sliding window
- **L2 normalization**: different from RMSNorm (uses rsqrt of sum-of-squares, not mean-of-squares)
- **Triangular solve**: forward substitution for the delta net's causal masking
- **Sum-rows reduction**: row-wise summation with different block sizes depending on SM count
- **F32 matrix multiplication**: state updates use F32×F32 GEMMs that normally go through cuBLAS (non-deterministic)
- **Multi-RoPE (MRoPE)**: position encoding with multiple rotation groups, each needing portable exponentiation

Each of these is a new divergence source that doesn't appear in standard transformer inference.

## Timeline of fixes

### Session 4: IQ4_XS dot products + infrastructure

Rebased deterministic patches to latest upstream (origin/master with Qwen3.5 architecture support). Wrote new deterministic dot products for IQ4_XS, Q5_K, Q8_0, and IQ4_NL weight types, each matching the GPU mmvq kernel's exact thread-to-block mapping and reduction order.

Key insight: the GPU mmvq kernel distributes each quantized block across multiple threads. For IQ4_XS, 16 threads per block handle 16 sub-blocks via table lookup (`kvalues_iq4nl`). For Q8_0, 4 threads per block each handle 8 bytes via `dp4a`. The CPU must replicate the exact same partitioning — computing partial sums over the same byte ranges, then reducing in the same order.

### Session 5: Attention V-accumulation + compile scope

**V accumulation order.** The GPU's `fattn-vec` kernel uses 4 warps × 32 threads with stride-4 V accumulation: 16 partial accumulators (4 warps × 4 K-position offsets), batch-32 max finding, online softmax with per-warp rescaling. The CPU used sequential position-by-position accumulation. This produced 1-ULP differences in 20% of attention output floats. Fix: complete rewrite of CPU attention to maintain 16 partial accumulators matching the GPU's warp structure.

**GGML_DETERMINISTIC compile scope.** The `#ifdef GGML_DETERMINISTIC` only reached the ggml-cuda target, not ggml-base. The `ggml_is_deterministic()` function in ggml.c fell back to a runtime env var check, returning false for GPU builds without the env var. This caused batch sizes > 8 to fall through to non-deterministic cuBLAS. Fix: `target_compile_definitions(ggml-base PUBLIC GGML_DETERMINISTIC)`.

### Session 6: Hybrid Mamba parity

Starting point: 13,256 / 15,246 tensors mismatched for Qwen3.5-4B-IQ4_XS.

**Fix 1: exp/expm1 unary ops.** Both `op_exp` and `op_expm1` used raw `expf` — missed when sigmoid and softplus were patched. Added `expf_det` guards on GPU and CPU. First tensor to match: `g_last_exp_t` (the decay factor exponentiation in gated delta net).

**Fix 2: gated_delta_net.cu portable expf.** Three `expf` calls in the fused GDN kernel replaced with `GDN_EXPF` macro (resolves to `expf_det` when `GGML_DETERMINISTIC`). Requires `#include "ggml-deterministic.h"`.

**Fix 3: SSM conv FMA matching.** The 1D convolution's inner loop `sumf += s[i] * c[i]` contracts to FMA on GPU via `--fmad=true`. CPU wasn't guaranteed to contract. Fix: explicit `fmaf(s[i], c[i], sumf)` in the deterministic path.

**Fix 4: L2 normalization.** GPU uses 32-thread strided accumulation with `__frsqrt_rn` for the scale. CPU used double-precision sequential sum with `1/sqrt`. Fix: CPU emulates GPU's 32-thread pattern with butterfly reduction, uses `(float)(1.0/sqrt((double)x))` for rsqrt.

**Fix 5: F32 matmul serialization.** The linear attention path computes state updates via F32×F32 GEMMs (e.g., `k^T × v` for outer products). These normally dispatch to cuBLAS which is non-deterministic. Fix: in deterministic mode, serialize F32 matmuls to single-column calls through the mmvf kernel, bypassing cuBLAS entirely. This was the highest-impact fix — ~6,500 fewer mismatches.

**Fix 6: Triangular solve.** GPU's warp-parallel forward substitution produces different FP results than CPU's sequential substitution due to different reduction order in the inner sum. Fix: new simple `solve_tri_f32_det` CUDA kernel with one thread per column, sequential row processing. CPU matches.

**Fix 7: MRoPE powf_int_det.** The `ggml_mrope_cache_init` function computed theta by iterative multiplication (`theta *= theta_scale`) which accumulates different FP error than GPU's `powf_int_det(theta_scale, n)`. Fix: CPU uses direct exponentiation matching GPU.

**Fix 8: Sum-rows reduction.** GPU's `reduce_rows_f32` uses variable block size (depends on `nrows/nsm`) with 8-way unrolled accumulation and block_reduce. CPU used `ggml_vec_sum_f32` which accumulates in double precision. Fix: new `sum_rows_f32_det` CUDA kernel with simple sequential float sum per row. CPU matches with `float sum += src_row[i]`. **This was the final fix** — achieving 0/15,246 mismatches.

**Fix 9: Q8_0 dot product structure.** The unsloth Qwen3.5-4B model uses Q8_0 for 48 tensors (vs heretic's 0). The original Q8_0 det dot product computed the full 32-byte block per "thread" then reduced across blocks. The GPU splits each block across 4 threads (8 bytes each via dp4a), producing different FP partial products. Fix: restructured CPU Q8_0 dot to match GPU's 128-thread layout with `qi=8, vdr=2, 4 threads per block, blocks_per_iter=32`.

### Verification

After all fixes: **15,246/15,246 tensors bitwise identical** for Qwen3.5-4B-IQ4_XS (both heretic imatrix and unsloth quants). **13,707/13,707** for Qwen3.5-27B-IQ4_XS. Multi-core CPU (8 threads) verified safe. Seeds verified portable across i9-9900K and GitHub Actions cloud runners.

## Reflections: why this worked

### Matching the GPU, not fighting it

The llama.cpp deterministic PR and the Thinking Machines blog post both approach determinism as "constrain the GPU." Disable fast-math, pad the KV cache, force specific kernel paths. We went the opposite direction: **leave the GPU mostly alone and make the CPU emulate its exact arithmetic.**

This meant reading CUDA kernels line by line and asking "what does `--fmad=true` actually do to this expression?" rather than "how do I make both sides use the same textbook algorithm?" The butterfly reductions, the stride-4 V accumulation, the 4-thread-per-block Q8_0 split — none of that is documented. It's emergent from the compiler and kernel design.

### The diagnostic made it tractable

The `det-diag` tool computes a 32-bit hash of every tensor during inference. Running on GPU and CPU and diffing the hashes turns "the output is wrong" into "tensor `node_91` at line 92 is a [64×64] matrix that first diverges here." You can binary-search the computation graph.

Each fix extends the matching prefix. Going from 13,256 → 6,558 → 5,904 → 5,688 → 0 was a directed walk through the divergence sources, not random probing.

### Tight iteration with full context

Having the complete history of every prior fix — which tensors matched, which diverged, what the root cause pattern looked like — was essential. Each new divergence is diagnosed faster because you've internalized the patterns: "this looks like FMA contraction" or "this looks like a reduction order issue." By session 6, identifying a new divergence source took minutes, not hours.

### The constraint that makes it novel

**GPU stays fast, CPU is the portable verifier.** This is the opposite of the zkML approach where the prover (GPU) must run inside a ZK circuit and is orders of magnitude slower. Our GPU inference runs at full speed with ~3-5% overhead. The CPU is slow but that's fine — it's a verifier, not a production system.

This framing — optimistic verification of ML inference — connects the systems work (making the bits match) to the theory of verifiable computation (what you can prove with deterministic intermediates).

## Toward verifiable inference

The deterministic execution trace we produce — tensor hashes at every layer — is a **witness** for a verifiable computation protocol:

1. **Optimistic verification** (what we have today): GPU produces output. CPU replays and checks. Sound, not zero-knowledge, not succinct, but practical and free.

2. **Structured verification** (future work): given claimed intermediate tensors `h₀, h₁, ..., h_L`, verify each layer independently. A Freivalds check verifies `h_{i+1} = W_i × h_i` with a single random inner product — O(n) instead of O(n²). Sum-check / GKR protocol makes this an interactive proof.

3. **Hybrid with ZK**: prove only the sampling step in ZK (tiny circuit: PRNG + argmax). The logits are deterministically reproducible so they don't need a ZK proof.

Our contribution is the prerequisite layer: reproducible intermediate values that make structured verification possible. The ZK community can build the efficient verifier on top.
