# Reproducible LLM Inference

**Prove that a specific LLM output is genuine — not cherry-picked or edited.**

Run a model on GPU with a seed, get interesting output, publish the seed.
Anyone with a CPU can verify they get the exact same output, bit-for-bit.

## The idea

LLMs are stochastic — given a temperature and seed, they sample from a probability distribution.
Normally GPU and CPU produce *different* outputs for the same seed because floating-point
arithmetic differs between CUDA and x86 at the level of individual bit-rounding decisions.

This project patches [llama.cpp](https://github.com/ggml-org/llama.cpp) to achieve
**bitwise-identical inference** across GPU and CPU. Every intermediate tensor — all 2462 of them
across 36 transformer layers — matches exactly. The same `(model, seed, prompt, temperature)`
produces the same tokens on a fast GPU and a slow CPU.

This enables a "dashcam for LLMs": grind seeds on GPU, find interesting outputs, publish the
seed. Anyone can verify on commodity hardware.

## Quick start

```bash
git clone https://github.com/amiller/llama_reproducible.git
cd llama_reproducible
./build.sh

# Download model (~2GB)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
  qwen2.5-3b-instruct-q4_k_m.gguf --local-dir models/

# Verify a seed
./verify.sh models/qwen2.5-3b-instruct-q4_k_m.gguf 2024 0.7 "Once upon a time"
# Expected: ", a little mouse used to go to the cinema every Saturday."
```

### Docker (no dependencies needed)

```bash
docker build -t det-verify .
docker run --rm -v $(pwd)/models:/models det-verify \
  -m /models/qwen2.5-3b-instruct-q4_k_m.gguf \
  -p "Once upon a time" -n 30 --seed 2024 --temp 0.7 --deterministic
```

## Example verified seeds

Model: `qwen2.5-3b-instruct-q4_k_m.gguf` (Q4_K_M quantization)

| Seed | Temp | Prompt | Output |
|------|------|--------|--------|
| 2024 | 0.7 | "Once upon a time" | ", a little mouse used to go to the cinema every Saturday. But when the pandemic hit, the cinema was closed." |
| 42 | 0.7 | "Once upon a time" | ", there was a boy named Ivan. Ivan loved to play in a forest and explore new places." |
| 777 | 0.7 | "Once upon a time" | ", there was a king who had a problem. He had a very powerful, but unfaithful, minister" |
| 9999 | 0.7 | "Once upon a time" | ", a king decided to build a tower of blocks. Each block was of the same size." |

These outputs are **deterministic** — run the same seed on any x86-64 machine with FMA support and you'll get the same text, token for token.

## What had to be fixed (18 phases)

Achieving bitwise CPU-GPU parity required fixing every source of floating-point divergence:

| Category | Issue | Fix |
|----------|-------|-----|
| **Transcendentals** | CUDA `expf`, `sinf`, `cosf`, `powf` use hardware approximations that differ from glibc by ±1 ULP | Portable polynomial implementations (`expf_det`, `sincosf_det`, `powf_int_det`) used on both sides |
| **FMA patterns** | NVCC auto-fuses multiply-add into FMA; GCC doesn't without `-mfma` | Explicit `fmaf()` calls + `-mfma` build flag |
| **Quantization** | x86 SIMD Q8_1 uses `roundf(x * 127/max)` (reciprocal); GPU uses `roundf(x / d)` (division) | Bypass x86 SIMD in deterministic mode |
| **RMSNorm** | Sequential reduction ≠ GPU's warp butterfly reduction; `rsqrtf()` is approximate (2 ULP) | Double-buffered butterfly + `__frsqrt_rn` on GPU |
| **Attention** | CPU uses online softmax (sequential); GPU processes tiles with parallel butterfly kqsum | Tile-based CPU attention matching GPU tile structure |
| **Normalization** | `VKQ * (1/S)` has double rounding vs GPU's `VKQ / S` | Direct division |
| **Batch size** | GPU JIT (ptxas) generates different SASS per `ncols_dst` template parameter | Force per-column mmvq processing |

## How it works

```
patches/           Patch files against llama.cpp (commit d092e268)
Dockerfile         Multi-stage build for CPU verifier
build.sh           Clone llama.cpp, apply patches, build
verify.sh          One-command seed verification
tools/             Additional verification tools
```

### Build from source

```bash
# Clone llama.cpp at the tested commit
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp
cd llama.cpp
git checkout d092e2682cc1db9f33b158b4378b448897b3096c

# Apply deterministic patches
git am ../patches/*.patch

# Build CPU verifier
cmake -B build \
  -DGGML_CUDA=OFF -DGGML_DETERMINISTIC=OFF \
  -DGGML_CPU_REPACK=OFF -DGGML_LLAMAFILE=OFF \
  -DGGML_NATIVE=OFF -DGGML_AVX2=OFF -DGGML_AVX=OFF \
  -DGGML_FMA=OFF -DGGML_F16C=OFF \
  -DCMAKE_C_FLAGS="-mfma" -DCMAKE_CXX_FLAGS="-mfma"
cmake --build build --target llama-det-diag -j$(nproc)
```

### GPU side (for generating reference outputs)

```bash
cmake -B build-gpu \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DGGML_DETERMINISTIC=ON
cmake --build build-gpu --target llama-det-diag -j$(nproc)
```

## GPU environment

The GPU reference output depends on:
- **CUDA toolkit version**: 12.8 (ptxas JIT generates architecture-specific SASS)
- **GPU architecture**: SM86 (RTX 3090). Other SM versions may produce different results
- **This patch set** against llama.cpp commit `d092e268`

The CPU verifier is architecture-independent (any x86-64 with FMA, i.e. Intel Haswell+ / AMD Piledriver+).

## Supported models

Currently verified for **Q4_K_M** quantization (Q4_K + Q6_K weight types). Other quant types (IQ4_XS, Q5_K) need additional deterministic dot product implementations.

## Related work

- **GPU determinism** (NVIDIA `CUBLAS_WORKSPACE_CONFIG`): same GPU → same results across runs. We go further: GPU → CPU parity.
- **Verifiable ML / ZKML** (EZKL, Modulus, Gensyn): cryptographic proofs that a model was evaluated correctly. Much heavier machinery; we use simple reproducibility.
- **Reproducible builds**: same concept applied to software compilation. We apply it to neural network inference.

## License

Patches are under the same license as llama.cpp (MIT).
