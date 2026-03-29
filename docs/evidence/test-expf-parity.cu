// Test expf parity between CUDA and glibc
// Compile: nvcc -o test-expf-parity test-expf-parity.cu -lm
// Run: ./test-expf-parity > gpu_expf.txt
// Then compare with CPU version

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

__global__ void compute_expf(const float *in, float *out, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) out[i] = expf(in[i]);
}

int main() {
    // Test range: values relevant to SiLU(-x) where x is typical layer activation
    // Typical range: -20 to 20
    const int N = 10000;
    float h_in[N], h_out[N];

    // Generate test values: -20 to 20 in steps, plus specific problematic values
    for (int i = 0; i < N; i++) {
        h_in[i] = -20.0f + 40.0f * i / (N - 1);
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    compute_expf<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Also compute CPU expf and compare
    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        float cpu_exp = expf(h_in[i]);
        uint32_t gpu_bits, cpu_bits;
        memcpy(&gpu_bits, &h_out[i], 4);
        memcpy(&cpu_bits, &cpu_exp, 4);
        if (gpu_bits != cpu_bits) {
            if (mismatches < 20) {
                printf("MISMATCH i=%d x=%.10e gpu_exp=%08x(%.10e) cpu_exp=%08x(%.10e) diff=%d\n",
                    i, h_in[i], gpu_bits, h_out[i], cpu_bits, cpu_exp, (int)gpu_bits - (int)cpu_bits);
            }
            mismatches++;
        }
    }
    printf("Total: %d / %d mismatches\n", mismatches, N);

    cudaFree(d_in);
    cudaFree(d_out);
}
