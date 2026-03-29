# Evidence: CUDA vs CPU floating-point divergence

These CUDA test programs prove specific sources of divergence between GPU and CPU inference. Each is a standalone `.cu` file you can compile with `nvcc` on a machine with a GPU.

## Files

### `test-expf-parity.cu`
**Proves:** CUDA's device `expf()` differs from glibc's `expf()` for ~30% of inputs in the SiLU-relevant range [-20, 20].

```
$ nvcc -arch=sm_86 test-expf-parity.cu -o test && ./test
expf mismatches: 2982/10000
```

Root cause: CUDA compiles `expf()` to `ex2.approx.ftz.f32` — a hardware lookup table instruction in the streaming multiprocessor. Not reproducible in software.

### `test-portable-expf.cu`
**Proves:** Our portable `expf_det()` implementation (Cody-Waite range reduction + Horner polynomial) gives 0 mismatches between GPU and CPU.

```
$ nvcc -arch=sm_86 -I ggml/include test-portable-expf.cu -o test && ./test
portable expf mismatches: 0/100000
```

### `test-expf-ptx-match.cu`
**Proves:** CUDA's `expf()` uses a specific pipeline: range reduce with Cody-Waite constants, then `ex2.approx.ftz.f32` hardware instruction, then reconstruct. Dumps PTX to verify.

### `test-expf-match.cu`
Exhaustive comparison of CUDA vs CPU `expf()` for every float in the SiLU range. Used to characterize the mismatch distribution.

### `test-expf-reverse.cu`
Reverse-engineers CUDA's internal polynomial coefficients by dumping range-reduced values and results. Used during development of `expf_det()`.

## How to run

Requires NVIDIA GPU + CUDA toolkit:

```bash
nvcc -arch=sm_86 -I /path/to/ggml/include <file>.cu -o test && ./test
```

Replace `sm_86` with your GPU's compute capability (e.g., `sm_89` for RTX 4090).
