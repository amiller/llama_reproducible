# Deterministic LLM Inference Verifier
# Produces bitwise-identical output to GPU for the same (model, seed, prompt, temp)
#
# Pinned for reproducibility:
#   - Ubuntu 22.04 base image pinned by digest
#   - llama.cpp pinned to exact commit
#   - Build flags frozen to produce identical FP arithmetic

FROM ubuntu:22.04@sha256:0e5e4a57c2499249aafc3b40fcd541e9a456aab7296681a3994d631587203f97 AS builder

ARG LLAMACPP_COMMIT=d092e2682cc1db9f33b158b4378b448897b3096c
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone llama.cpp at pinned commit
RUN git clone https://github.com/ggml-org/llama.cpp.git /src \
    && cd /src && git checkout ${LLAMACPP_COMMIT}

# Apply deterministic parity patches
COPY patches/ /patches/
RUN cd /src && git am /patches/*.patch

# CPU deterministic build
# -mfma: hardware FMA matching NVCC auto-contraction
# All other GGML SIMD features off to prevent non-deterministic paths
RUN cd /src && cmake -B build \
    -DGGML_CUDA=OFF \
    -DGGML_DETERMINISTIC=OFF \
    -DGGML_CPU_REPACK=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX2=OFF \
    -DGGML_AVX=OFF \
    -DGGML_FMA=OFF \
    -DGGML_F16C=OFF \
    -DCMAKE_C_FLAGS="-mfma" \
    -DCMAKE_CXX_FLAGS="-mfma" \
    && cmake --build build --target llama-det-diag -j$(nproc)

# Minimal runtime (also pinned)
FROM ubuntu:22.04@sha256:0e5e4a57c2499249aafc3b40fcd541e9a456aab7296681a3994d631587203f97

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libcurl4 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/build/bin/llama-det-diag /usr/local/bin/

ENTRYPOINT ["llama-det-diag"]
