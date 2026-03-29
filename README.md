# llama_reproducible

**A dashcam for your local LLM.**

You run a model all day. Occasionally it says something wild — hilarious, profound, eerily specific. Normally you'd screenshot it and move on. With reproducible inference, you just save the seed. Anyone with the same model can replay the exact moment, token for token.

## How it works

LLMs with temperature sampling are stochastic — a seed determines which path through the probability space you take. But normally, GPU and CPU produce *different* outputs for the same seed because floating-point arithmetic differs at the bit level between CUDA and x86.

This project patches [llama.cpp](https://github.com/ggml-org/llama.cpp) to make GPU and CPU inference **bitwise identical**. Every intermediate tensor — all 2462 across 36 transformer layers — matches exactly. The same `(model, seed, prompt, temperature)` produces the same tokens whether you run it on a fast GPU or a slow laptop.

A seed becomes a tiny pointer into the space of everything a model could say. The model itself is the shared codebook.

## Try it

```bash
git clone https://github.com/amiller/llama_reproducible.git
cd llama_reproducible
./build.sh

# Download the model (~2GB)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
  qwen2.5-3b-instruct-q4_k_m.gguf --local-dir models/

# Replay a seed
./verify.sh models/qwen2.5-3b-instruct-q4_k_m.gguf 2024 0.7 "Once upon a time"
```

### Docker (nothing to install)

```bash
docker build -t llama-replay .
docker run --rm -v $(pwd)/models:/models llama-replay \
  -m /models/qwen2.5-3b-instruct-q4_k_m.gguf \
  -p "Once upon a time" -n 30 --seed 2024 --temp 0.7 --deterministic
```

## Example seeds

Model: `qwen2.5-3b-instruct-q4_k_m.gguf` — temp 0.7, prompt "Once upon a time"

| Seed | Output |
|------|--------|
| 2024 | ", a little mouse used to go to the cinema every Saturday. But when the pandemic hit, the cinema was closed." |
| 42 | ", there was a boy named Ivan. Ivan loved to play in a forest and explore new places." |
| 777 | ", there was a king who had a problem. He had a very powerful, but unfaithful, minister" |
| 9999 | ", a king decided to build a tower of blocks. Each block was of the same size." |

Pick a seed, run it on any modern x86 laptop. Same text comes out.

## What's inside

```
patches/        3 patch files against llama.cpp (commit d092e268)
Dockerfile      Pinned build for reproducibility
build.sh        Clone llama.cpp, apply patches, build
verify.sh       Replay a seed
docs/           Investigation notes, evidence, GPU setup guide
```

## What had to be fixed

18 separate fixes across the GPU and CPU codepaths. The short version: every math function on a GPU (`expf`, `sinf`, `cosf`, `powf`, `rsqrtf`) produces slightly different bits than the same function on a CPU. Every compiler (NVCC vs GCC) makes different decisions about when to fuse `a*b+c` into a single instruction. The GPU's quantization code rounds differently than the CPU's x86 SIMD version. The attention mechanism processes data in a different order.

Each of these differences is ±1 in the last bit of a 32-bit float. But they compound through 36 layers of a neural network until the outputs diverge completely.

See [docs/investigation.md](docs/investigation.md) for the full technical story.

## GPU setup

The GPU side generates the "reference" outputs at ~200 tokens/sec. The CPU replays them at ~2 tokens/sec. See [docs/gpu-setup.md](docs/gpu-setup.md) for the exact CUDA/driver/GPU environment used.

The CPU replay works on any modern x86 laptop (Intel ~2013+ / AMD ~2012+). The Docker build pins everything for full reproducibility.

## Current scope

- Verified for **Q4_K_M** quantization (most common GGUF format)
- Tested with qwen2.5-3b — small enough to run on any machine
- IQ4_XS and MoE models need additional work (see docs)

## License

Patches under the same license as llama.cpp (MIT).
