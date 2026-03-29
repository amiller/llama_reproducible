// Reverse-engineer CUDA's expf by dumping range-reduced values and results
// to find the exact polynomial coefficients
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

// Check: does CUDA expf match exp2f(x * log2e)?
__global__ void test_exp2_path(const float *in, float *out_expf, float *out_exp2, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        out_expf[i] = expf(in[i]);
        out_exp2[i] = exp2f(in[i] * 1.442695040888963f);
    }
}

int main() {
    const int N = 100000;
    float *h_in = new float[N];
    float *h_expf = new float[N];
    float *h_exp2 = new float[N];

    for (int i = 0; i < N; i++)
        h_in[i] = -20.0f + 40.0f * i / (N - 1);

    float *d_in, *d_expf, *d_exp2;
    cudaMalloc(&d_in, N*4); cudaMalloc(&d_expf, N*4); cudaMalloc(&d_exp2, N*4);
    cudaMemcpy(d_in, h_in, N*4, cudaMemcpyHostToDevice);

    test_exp2_path<<<(N+255)/256, 256>>>(d_in, d_expf, d_exp2, N);
    cudaMemcpy(h_expf, d_expf, N*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_exp2, d_exp2, N*4, cudaMemcpyDeviceToHost);

    int match = 0;
    for (int i = 0; i < N; i++) {
        uint32_t a, b;
        memcpy(&a, &h_expf[i], 4);
        memcpy(&b, &h_exp2[i], 4);
        if (a == b) match++;
    }
    printf("expf(x) == exp2f(x*log2e): %d / %d\n", match, N);

    // Also check: expf(x) == ldexpf(exp2f(frac), int_part)?
    // Try: expf(x) decomposed as:
    //   t = x * log2e
    //   n = rintf(t)
    //   f = t - n
    //   result = ldexpf(exp2f(f), (int)n)
    float *h_decomp = new float[N];
    for (int i = 0; i < N; i++) {
        float t = h_in[i] * 1.442695040888963f;
        float n = rintf(t);
        // But we need exp2f(f) computed on GPU...
    }

    // Check if CUDA's expf is literally just calling __internal_expf or similar
    // by checking PTX output
    printf("\nTo check PTX: nvcc -ptx test-expf-reverse.cu\n");
    printf("Look for ex2.approx.f32 vs __nv_expf\n");

    delete[] h_in; delete[] h_expf; delete[] h_exp2; delete[] h_decomp;
    cudaFree(d_in); cudaFree(d_expf); cudaFree(d_exp2);
}
