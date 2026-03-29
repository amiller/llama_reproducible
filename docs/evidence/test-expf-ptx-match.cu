// Replicate the exact PTX sequence of CUDA's expf on CPU
// Key: the hardware ex2.approx.ftz.f32 instruction
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

// Exact replica of the PTX from nvcc's expf() compilation
// Works on CPU by using exp2f() for the ex2.approx step
static inline float expf_cuda_ptx(float x) {
    // PTX constants (from the disassembly)
    const float c1 = 0.5f;                    // 0f3F000000
    const float c2 = 5.7314485e-03f;          // 0f3BBB989D — actually let me decode properly
    const float c3 = 252.0f;                  // 0f437C0000
    const float c4 = 12582913.0f;             // 0f4B400001 — rounding magic
    const float c5 = -12582784.0f + 127.0f;   // 0fCB40007F = -12582784 + ... decode properly
    const float log2e_hi = 1.44269502163f;     // 0f3FB8AA3B
    const float log2e_lo = 1.92596299e-8f;     // 0f32A57060

    // PTX sequence:
    // f4 = fma.rn(x, c2, c1)          -- f4 = x * 5.73e-3 + 0.5
    // f5 = cvt.sat(f4)                -- clamp to [0,1]
    // f8 = fma.rm(f5, c3, c4)         -- f8 = f5 * 252 + 12582913 (round-to-minus-inf)
    // f9 = f8 + c5                    -- f9 = f8 - 12582784 + 127... this extracts integer part
    // f10 = -f9                       -- negate
    // f12 = fma.rn(x, log2e_hi, f10)  -- f12 = x*log2e_hi - f9 (reduced argument)
    // f14 = fma.rn(x, log2e_lo, f12)  -- f14 = x*log2e_lo + f12 (Cody-Waite correction)
    // r1 = bitcast(f8)                -- integer bits of f8
    // r2 = r1 << 23                   -- shift to exponent position (2^n scaling)
    // f15 = bitcast(r2)               -- 2^n as float
    // f16 = ex2.approx.ftz(f14)       -- hardware 2^f14
    // f17 = f16 * f15                 -- result = 2^f14 * 2^n

    // Replicate:
    float f4 = fmaf(x, 5.7314485e-03f, 0.5f);
    float f5 = fmaxf(0.0f, fminf(1.0f, f4)); // cvt.sat

    // fma.rm (round toward minus infinity) — use floor-based approach
    // Actually fma.rm is FMA with round-to-negative-infinity mode
    // On CPU we can't easily control rounding mode per-instruction
    // Let's try: the purpose is to compute floor(f5 * 252 + 0.5) basically
    // Actually c4 = 12582913 = 2^23 * 1.5 + 1 — this is the "magic number" rounding trick
    // fma.rm(f5, 252, 12582913) with round-down gives the integer part embedded in float bits

    // For now, let's try computing n = floor(x * log2e + 0.5) = round(x * log2e)
    // and r = x * log2e - n using Cody-Waite

    // Actually, let me decode the PTX more carefully using hex constants
    uint32_t tmp;

    // c2 = 0x3BBB989D
    tmp = 0x3BBB989D; float c2_exact; memcpy(&c2_exact, &tmp, 4);

    // c5 = 0xCB40007F
    tmp = 0xCB40007F; float c5_exact; memcpy(&c5_exact, &tmp, 4);

    f4 = fmaf(x, c2_exact, 0.5f);
    f5 = fmaxf(0.0f, fminf(1.0f, f4));

    // fma.rm — round to minus infinity
    // We need to emulate this. fesetround(FE_DOWNWARD) is thread-unsafe.
    // Alternative: use the bit-trick directly
    // The magic number trick: adding 2^23*1.5 to a float in [0, 252] and reading the mantissa bits
    // gives the rounded integer value.
    // fma.rm(f5, 252, 12582913) with round-down:
    float f8_product = f5 * 252.0f + 12582913.0f;
    // With round-to-nearest, this might differ from round-to-minus-inf by 1
    // For now, approximate:
    float f8 = f8_product; // TODO: need round-toward-minus-inf

    float f9 = f8 + c5_exact;
    float f10 = -f9;

    // Cody-Waite reduction
    tmp = 0x3FB8AA3B; float log2e_hi_exact; memcpy(&log2e_hi_exact, &tmp, 4);
    tmp = 0x32A57060; float log2e_lo_exact; memcpy(&log2e_lo_exact, &tmp, 4);

    float f12 = fmaf(x, log2e_hi_exact, f10);
    float f14 = fmaf(x, log2e_lo_exact, f12);

    // 2^n via bit manipulation
    uint32_t r1;
    memcpy(&r1, &f8, 4);
    uint32_t r2 = r1 << 23;
    float f15;
    memcpy(&f15, &r2, 4);

    // ex2.approx.ftz = exp2f(f14) — this is the hardware instruction
    // On CPU, use exp2f which should be very close
    float f16 = exp2f(f14);

    return f16 * f15;
}

__global__ void compute_native(const float *in, float *out, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) out[i] = expf(in[i]);
}

int main() {
    const int N = 100000;
    float *h_in = new float[N], *h_gpu = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = -20.0f + 40.0f * i / (N-1);

    float *d_in, *d_out;
    cudaMalloc(&d_in, N*4); cudaMalloc(&d_out, N*4);
    cudaMemcpy(d_in, h_in, N*4, cudaMemcpyHostToDevice);
    compute_native<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaMemcpy(h_gpu, d_out, N*4, cudaMemcpyDeviceToHost);

    int exact = 0, off1 = 0;
    for (int i = 0; i < N; i++) {
        float cpu = expf_cuda_ptx(h_in[i]);
        uint32_t a, b;
        memcpy(&a, &cpu, 4); memcpy(&b, &h_gpu[i], 4);
        if (a == b) exact++;
        else if (abs((int)a-(int)b) <= 1) off1++;
    }
    printf("PTX-replica exact match: %d / %d\n", exact, N);
    printf("Off by 1 ULP: %d\n", off1);
    printf("Off by >1 ULP: %d\n", N - exact - off1);

    delete[] h_in; delete[] h_gpu;
    cudaFree(d_in); cudaFree(d_out);
}
