# llama_reproducible

**A dashcam for your local LLM.**

You run a model all day. Occasionally it says something wild — hilarious, profound, eerily specific. Normally you'd screenshot it and move on. With reproducible inference, you just save the seed. Anyone with the same model can replay the exact moment, token for token.

## How it works

LLM text generation is **pseudorandom** — at each step the model produces a probability distribution over tokens, and a [PRNG](https://en.wikipedia.org/wiki/Pseudorandom_number_generator) seeded by a fixed integer decides which token gets sampled. Same seed, same sequence of random choices, same output text. The model weights and the seed together determine a specific trajectory through the space of possible completions.

The problem: this only works if the model produces the **exact same probability distribution** on every machine. In practice, GPU and CPU produce different distributions for the same input because [floating-point arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) differs at the bit level between CUDA and x86. A ±1 difference in the last bit of a 32-bit float compounds through 36 transformer layers until the outputs diverge completely.

This project patches [llama.cpp](https://github.com/ggml-org/llama.cpp) to make GPU and CPU inference **bitwise identical** — every intermediate tensor (all 2462 of them) matches exactly. The same `(model, seed, prompt, temperature)` produces the same tokens whether you run it on a fast GPU or a slow laptop.

A seed becomes a tiny pointer into the space of everything a model could say. The model itself is the shared codebook.

### References

Direct lineage:
- [llama.cpp deterministic PR #16016](https://github.com/ggml-org/llama.cpp/pull/16016) — the GPU-to-GPU determinism work our patches build on top of.
- [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (Horace He, Thinking Machines Lab) — identifies batch invariance as the root cause of non-determinism. Inspired the llama.cpp PR.

Floating-point background:
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) — why `a*b+c` on two different chips can give different bits.
- [IEEE 754 FMA](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation#Fused_multiply%E2%80%93add) — the fused multiply-add instruction at the heart of most divergences.

ZKML (different approach, different tradeoffs):
- [zkLLM](https://arxiv.org/abs/2404.16109), [ZKML](https://ddkang.github.io/papers/2024/zkml-eurosys.pdf) (Kang et al.), [survey](https://arxiv.org/html/2502.18535v1), [Ingonyama](https://www.ingonyama.com/blog/unleashing-secure-ai) — cryptographic proofs of inference. ZKML lets a verifier check a ~200KB proof without running the model, but the prover must run inference inside a ZK circuit (orders of magnitude slower than native). You have to decide *before* inference that you want a proof. Our approach is the opposite: run your model normally at full speed, and every inference is automatically replayable because determinism is a property of the execution, not something bolted on. Verification costs 30 seconds of CPU time instead of being instant, but recording costs nothing and is always on — like a dashcam's rolling buffer.

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

Model: [`qwen2.5-3b-instruct-q4_k_m.gguf`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) (rev [`7dabda4`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/tree/7dabda4d13d513e3e842b20f0d435c732f172cbe), SHA256: `626b4a6678b86442...`) — temp 0.7, prompt "Once upon a time"

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

## Background: deterministic ≠ portable

llama.cpp already has a [`--deterministic` mode](https://github.com/ggml-org/llama.cpp) (our patches build on top of [this work](https://github.com/ggml-org/llama.cpp/tree/deterministic)). It guarantees **GPU-to-GPU reproducibility** — the same GPU with the same driver produces the same output across runs. This is useful but doesn't help if you want to replay a seed on a different machine.

**Portable reproducibility** — getting a CPU to match a specific GPU — is a harder problem. The GPU is the fast "daily driver" that generates output at ~200 tok/s. The CPU is the slow "shadow" (~2 tok/s) that anyone can run to replay the same seed. The goal is not to make the CPU fast, just to make it agree.

## Design constraints

- **GPU stays fast.** We don't cripple the GPU kernel. Most fixes are on the CPU side, making it emulate the GPU's arithmetic. The few GPU-side changes (`__frsqrt_rn` instead of approximate `rsqrtf`, portable `expf_det` in SiLU) cost ~3-5% overhead.
- **Specific environment.** We targeted one setup: **RTX 3090 (SM86) + CUDA 12.8**. Different GPUs or CUDA versions may produce different reference outputs — the JIT compiler generates different machine code per architecture. Extending to other GPUs means re-verifying (and possibly re-tuning) the CPU shadow.
- **One quant format.** Verified for Q4_K_M (the most common GGUF quantization). Other formats need additional work.

## What had to be fixed

The existing `--deterministic` branch handles GPU-to-GPU consistency (disabling fast-math, padding the KV cache, etc.). On top of that, we fixed 18 additional sources of CPU-GPU divergence:

**Portable math functions.** CUDA's `expf`, `sinf`, `cosf`, `powf`, `rsqrtf` all use hardware approximations that differ from the CPU's glibc by ±1 in the last bit. We wrote portable polynomial implementations (`expf_det`, `sincosf_det`, `powf_int_det`) used on both sides. On the GPU these replace only the non-deterministic functions (SiLU activation, RoPE, RMSNorm, attention softmax) — the quantized matmul kernels are untouched. Overhead: **~3-5%** on GPU, unmeasurable on the memory-bound inference kernels.

**FMA pattern matching.** NVCC silently turns `a*b+c` into a fused multiply-add. GCC doesn't unless you give it `-mfma`. We traced the GPU's exact fusion decisions using PTX disassembly (`nvcc --ptx`) and SASS disassembly (`cuobjdump --dump-sass`) to match every multiply-add pattern on the CPU.

**Attention tile structure.** The GPU processes key-value positions in parallel tiles with warp-level butterfly reductions. The CPU used sequential online softmax. We rewrote the CPU attention to match the GPU's tile structure, including the butterfly reduction order for the softmax denominator.

Each individual divergence is ±1 in the last bit of a 32-bit float. But they compound through 36 layers until the outputs diverge completely.

See [docs/investigation.md](docs/investigation.md) for the full technical story, and [docs/gpu-setup.md](docs/gpu-setup.md) for the exact CUDA/driver/GPU environment.

## Current scope

- Built on top of the [llama.cpp deterministic branch](https://github.com/ggml-org/llama.cpp/tree/deterministic) (commit `d092e268`)
- Verified for **Q4_K_M** quantization on **RTX 3090 / CUDA 12.8**
- Tested with qwen2.5-3b — small enough to run on any machine
- IQ4_XS, MoE models, and other GPU architectures need additional work

## License

Patches under the same license as llama.cpp (MIT).
