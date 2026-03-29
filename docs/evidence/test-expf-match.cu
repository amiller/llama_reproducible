// Exhaustive expf comparison for the SiLU-relevant range
// Output: binary dump of CUDA expf results for every float in [-20, 20]
// We'll use this to build a matching CPU implementation
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

// Portable CUDA-matching expf using Cody-Waite + polynomial
// Based on NVIDIA's documented algorithm
static inline float expf_cuda_match(float x) {
    // Range reduction: x = n*ln2 + r, |r| <= ln2/2
    // Using Cody-Waite for exact subtraction
    const float LOG2EF = 1.4426950408889634f;   // 1/ln(2)
    const float LN2_HI = 0.693145751953125f;     // ln(2) high part (exact in float)
    const float LN2_LO = 1.428606765330187e-06f;  // ln(2) low part

    float n = rintf(x * LOG2EF);  // round to nearest integer
    float r = x - n * LN2_HI;
    r = r - n * LN2_LO;

    // Minimax polynomial for exp(r) on [-ln2/2, ln2/2]
    // Coefficients matching CUDA's implementation
    float p = 1.0f;
    p = fmaf(p, r, 1.0f);           // 1 + r
    p = fmaf(0.5f, r * r, p - 1.0f) + 1.0f;  // hmm this isn't right

    // Actually, standard approach:
    // exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120
    // Using Horner's form with FMA
    float r2 = r * r;
    p = fmaf(1.9875691500e-4f, r, 1.3981999507e-3f);
    p = fmaf(p, r, 8.3334519073e-3f);
    p = fmaf(p, r, 4.1665795894e-2f);
    p = fmaf(p, r, 1.6666665459e-1f);
    p = fmaf(p, r, 5.0000001201e-1f);
    p = fmaf(p, r2, r);
    p = p + 1.0f;

    // Reconstruct: exp(x) = p * 2^n
    int ni = (int)n;
    // Use bit manipulation to multiply by 2^n
    uint32_t bits;
    memcpy(&bits, &p, 4);
    bits += (uint32_t)ni << 23;  // add n to exponent
    memcpy(&p, &bits, 4);

    return p;
}

__global__ void compute_expf_gpu(const float *in, float *out, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) out[i] = expf(in[i]);
}

int main() {
    const int N = 100000;
    float h_in[N], h_gpu[N];

    for (int i = 0; i < N; i++)
        h_in[i] = -20.0f + 40.0f * i / (N - 1);

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    compute_expf_gpu<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaMemcpy(h_gpu, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    int glibc_match = 0, cuda_match_match = 0, both_mismatch = 0;
    for (int i = 0; i < N; i++) {
        float cpu_std = expf(h_in[i]);
        float cpu_custom = expf_cuda_match(h_in[i]);
        uint32_t gb, sb, cb;
        memcpy(&gb, &h_gpu[i], 4);
        memcpy(&sb, &cpu_std, 4);
        memcpy(&cb, &cpu_custom, 4);

        if (gb == sb) glibc_match++;
        if (gb == cb) cuda_match_match++;
        if (gb != sb && gb != cb) both_mismatch++;
    }
    printf("glibc matches GPU: %d / %d\n", glibc_match, N);
    printf("custom matches GPU: %d / %d\n", cuda_match_match, N);
    printf("neither matches: %d\n", both_mismatch);

    cudaFree(d_in);
    cudaFree(d_out);
}
