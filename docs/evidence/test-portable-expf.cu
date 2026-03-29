// Portable expf using only FMA + basic IEEE ops — same result on CPU and GPU
// Based on Cody-Waite range reduction + Remez minimax polynomial
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

// Portable expf: identical results on CPU and GPU when compiled with FMA
// Uses Cody-Waite range reduction + degree-6 minimax polynomial
#ifdef __CUDA_ARCH__
#define PORTABLE_FMA(a,b,c) __fmaf_rn(a,b,c)
#define PORTABLE_RINTF(x) rintf(x)
#else
#define PORTABLE_FMA(a,b,c) fmaf(a,b,c)
#define PORTABLE_RINTF(x) rintf(x)
#endif

static __host__ __device__ float expf_portable(float x) {
    // Clamp to avoid overflow/underflow
    x = fmaxf(x, -87.3f);
    x = fminf(x, 88.7f);

    // Range reduction: exp(x) = 2^n * exp(r)
    // n = round(x / ln2), r = x - n*ln2
    const float log2e = 1.442695040888963f;
    const float ln2_hi = 0.693145751953125f;     // 24 bits, exact in float
    const float ln2_lo = 1.42860676533018e-06f;  // remainder

    float n = PORTABLE_RINTF(x * log2e);
    float r = PORTABLE_FMA(n, -ln2_hi, x);     // r = x - n*ln2_hi (exact with FMA)
    r = PORTABLE_FMA(n, -ln2_lo, r);            // r = r - n*ln2_lo

    // Minimax polynomial for exp(r)-1 on [-ln2/2, ln2/2], degree 5
    // exp(r) ≈ 1 + r + c2*r^2 + c3*r^3 + c4*r^4 + c5*r^5
    // Horner form: p = ((((c5*r + c4)*r + c3)*r + c2)*r + 1)*r + 1
    float p = 1.9875691500e-4f;
    p = PORTABLE_FMA(p, r, 1.3981999507e-3f);
    p = PORTABLE_FMA(p, r, 8.3334519073e-3f);
    p = PORTABLE_FMA(p, r, 4.1665795894e-2f);
    p = PORTABLE_FMA(p, r, 1.6666665459e-1f);
    p = PORTABLE_FMA(p, r, 5.0000001201e-1f);
    p = PORTABLE_FMA(p, r * r, r);  // p = p*r^2 + r
    p = p + 1.0f;                   // exp(r) ≈ p

    // Reconstruct: exp(x) = p * 2^n via exponent manipulation
    int ni = (int)n;
    uint32_t bits;
    memcpy(&bits, &p, sizeof(bits));
    bits += (uint32_t)(ni) << 23;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

__global__ void compute_portable_gpu(const float *in, float *out, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) out[i] = expf_portable(in[i]);
}

__global__ void compute_expf_gpu(const float *in, float *out, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) out[i] = expf(in[i]);
}

int main() {
    const int N = 100000;
    float *h_in = new float[N];
    float *h_gpu_native = new float[N];
    float *h_gpu_portable = new float[N];

    for (int i = 0; i < N; i++)
        h_in[i] = -20.0f + 40.0f * i / (N - 1);

    float *d_in, *d_out1, *d_out2;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out1, N * sizeof(float));
    cudaMalloc(&d_out2, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    compute_expf_gpu<<<(N+255)/256, 256>>>(d_in, d_out1, N);
    compute_portable_gpu<<<(N+255)/256, 256>>>(d_in, d_out2, N);
    cudaMemcpy(h_gpu_native, d_out1, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gpu_portable, d_out2, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare: CPU portable vs GPU portable (should be identical if FMA matches)
    // And: CPU portable vs GPU native (how close?)
    int cpu_gpu_portable_match = 0, cpu_gpu_native_match = 0, gpu_port_gpu_native = 0;
    int cpu_portable_off_by_1 = 0;
    for (int i = 0; i < N; i++) {
        float cpu_port = expf_portable(h_in[i]);
        uint32_t cp, gp, gn;
        memcpy(&cp, &cpu_port, 4);
        memcpy(&gp, &h_gpu_portable[i], 4);
        memcpy(&gn, &h_gpu_native[i], 4);

        if (cp == gp) cpu_gpu_portable_match++;
        if (cp == gn) cpu_gpu_native_match++;
        if (gp == gn) gpu_port_gpu_native++;
        if (abs((int)cp - (int)gn) <= 1) cpu_portable_off_by_1++;
    }
    printf("CPU_portable == GPU_portable: %d / %d (tests FMA parity)\n", cpu_gpu_portable_match, N);
    printf("CPU_portable == GPU_native:   %d / %d (our goal)\n", cpu_gpu_native_match, N);
    printf("GPU_portable == GPU_native:   %d / %d (polynomial quality)\n", gpu_port_gpu_native, N);
    printf("CPU_portable within 1 ULP of GPU_native: %d / %d\n", cpu_portable_off_by_1, N);

    delete[] h_in;
    delete[] h_gpu_native;
    delete[] h_gpu_portable;
    cudaFree(d_in); cudaFree(d_out1); cudaFree(d_out2);
}
