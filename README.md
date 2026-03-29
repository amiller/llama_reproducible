# llama_reproducible

**A dashcam for your local LLM.**

You run a model all day. Occasionally it says something wild — hilarious, profound, eerily specific. Normally you'd screenshot it and move on. With reproducible inference, you just save the seed. Anyone with the same model can replay the exact moment, token for token.

## How it works

LLM text generation is **pseudorandom** — at each step the model produces a probability distribution over tokens, and a [PRNG](https://en.wikipedia.org/wiki/Pseudorandom_number_generator) seeded by a fixed integer decides which token gets sampled. Same seed, same sequence of random choices, same output text. The model weights and the seed together determine a specific trajectory through the space of possible completions.

The problem: this only works if the model produces the **exact same probability distribution** on every machine. In practice, GPU and CPU produce different distributions for the same input because [floating-point arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) differs at the bit level between CUDA and x86. A ±1 difference in the last bit of a 32-bit float compounds through 36 transformer layers until the outputs diverge completely.

This project patches [llama.cpp](https://github.com/ggml-org/llama.cpp) to make GPU and CPU inference **bitwise identical** — every intermediate tensor (all 2462 of them) matches exactly. The same `(model, seed, prompt, temperature)` produces the same tokens whether you run it on a fast GPU or a slow laptop.

A seed becomes a tiny pointer into the space of everything a model could say. The model itself is the shared codebook.

### Related work

- [NVIDIA deterministic computing](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility) — same GPU → same results across runs. We go further: GPU ↔ CPU parity.
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) — why `a*b+c` on two different chips can give different bits.
- [IEEE 754 FMA](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation#Fused_multiply%E2%80%93add) — the fused multiply-add instruction at the heart of most divergences.
- [Verifiable ML / ZKML](https://blog.ezkl.xyz/post/verify/) — cryptographic proofs of inference. Heavier machinery solving a different problem; we use simple reproducibility.

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
