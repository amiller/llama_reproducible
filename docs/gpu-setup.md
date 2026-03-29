# GPU Reference Setup

This documents the exact environment used to generate the GPU reference outputs. Anyone with a similar setup can reproduce the GPU side and verify their CPU verifier matches.

## Hardware

- **GPU:** NVIDIA GeForce RTX 3090 (24GB VRAM, SM86 / compute capability 8.6)
- **CPU:** Intel Core i9-9900K
- **RAM:** sufficient for the 2GB model

## Software

| Component | Version | Notes |
|-----------|---------|-------|
| OS | Linux Mint 21.3 (Ubuntu 22.04 base) | Any Ubuntu 22.04 derivative should work |
| Kernel | 6.8.0-106-generic | |
| NVIDIA Driver | 580.126.09 | |
| CUDA Toolkit | 12.8 (build cuda_12.8.r12.8/compiler.35583870_0) | Installed via .run installer with `--toolkit` flag, NOT apt |
| GCC | 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04.3) | |
| llama.cpp | commit `d092e268` + patches | |

## CUDA installation

**Important:** Do NOT install CUDA via `apt install nvidia-cuda-toolkit`. Use the .run installer:

```bash
# Download CUDA 12.8 from NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run

# Install toolkit only (don't touch the driver)
sudo sh cuda_12.8.0_570.86.10_linux.run --toolkit --silent --override

# Verify
/usr/local/cuda-12.8/bin/nvcc --version
```

## GPU build

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout d092e2682cc1db9f33b158b4378b448897b3096c

# Apply patches from this repo
git am /path/to/llama_reproducible/patches/*.patch

# Build GPU deterministic
cmake -B build-det-gpu \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
    -DGGML_DETERMINISTIC=ON \
    -DGGML_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TESTS=ON
cmake --build build-det-gpu --target llama-det-diag -j$(nproc)
```

## Running the GPU reference

```bash
# Stop any running llama-server (it grabs VRAM)
sudo systemctl stop llama-server 2>/dev/null

# Generate reference tokens
build-det-gpu/bin/llama-det-diag \
    -m /path/to/qwen2.5-3b-instruct-q4_k_m.gguf \
    -p "Once upon a time" \
    -n 30 --seed 2024 --temp 0.7 \
    2>&1 | grep "^Token"
```

Expected output:
```
Token 0: id=11 ','
Token 1: id=264 ' a'
Token 2: id=2632 ' little'
Token 3: id=8644 ' mouse'
Token 4: id=1483 ' used'
Token 5: id=311 ' to'
Token 6: id=728 ' go'
Token 7: id=311 ' to'
Token 8: id=279 ' the'
Token 9: id=33192 ' cinema'
...
```

## Tensor-level verification

To verify ALL tensors match between GPU and CPU:

```bash
# GPU hashes
build-det-gpu/bin/llama-det-diag \
    -m model.gguf -p "The quick brown fox" -n 1 --seed 42 --temp 0 \
    2>&1 | grep "^T[0-9]" > /tmp/gpu.txt

# CPU hashes
build-det-cpu/bin/llama-det-diag \
    -m model.gguf -p "The quick brown fox" -n 1 --seed 42 --temp 0 --deterministic \
    2>&1 | grep "^T[0-9]" > /tmp/cpu.txt

# Compare
diff /tmp/gpu.txt /tmp/cpu.txt && echo "ALL TENSORS MATCH"
```

## What changes the GPU output

The GPU output is a function of:

1. **CUDA toolkit version** — ptxas (the GPU JIT) generates different SASS per version
2. **GPU architecture** — SM86 SASS differs from SM89, SM90, etc.
3. **llama.cpp commit + patches** — the source code
4. **Model file** — the quantized weights (verified by SHA256)

If you have the same CUDA 12.8 + SM86 GPU + this exact code, you should get identical outputs. Different CUDA versions or GPU architectures may need re-verification of the GPU side (the CPU side is architecture-independent).

## Performance impact

GPU overhead from deterministic mode:
- Removing `-use_fast_math`: ~1-3% slower
- `expf_det` in SiLU/attention: negligible (memory-bound kernels)
- `__frsqrt_rn` in RMSNorm: negligible
- Per-column mmvq: slightly slower for batch>1 prompt eval (splits into N kernel launches)
- Total: **~3-5% slower** than non-deterministic GPU inference

## Complete environment fingerprint

```
GPU:           NVIDIA GeForce RTX 3090 (GA102, SM86, 24GB)
Driver:        580.126.09
CUDA Toolkit:  12.8 (build cuda_12.8.r12.8/compiler.35583870_0)
ptxas:         V12.8.93
GCC:           11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04.3)
OS:            Linux Mint 21.3 (Ubuntu 22.04 / Jammy base)
Kernel:        6.8.0-106-generic
CPU:           Intel Core i9-9900K @ 3.60GHz (Coffee Lake, FMA3)
```

The driver version matters because the NVIDIA driver includes `ptxas` (the JIT
compiler that converts PTX to SASS at runtime). Different driver versions may
produce different SASS for the same PTX, potentially changing rounding behavior.

For best reproducibility, match the CUDA toolkit version (12.8) and GPU
architecture (SM86). The driver version is less critical since our patches use
`__fmul_rn`/`__fmaf_rn`/`__frsqrt_rn` intrinsics that constrain the JIT's
optimization freedom. But it has not been tested with other driver versions.
