#!/bin/bash
# Build the CPU deterministic verifier from source (no Docker needed)
set -euo pipefail

LLAMACPP_COMMIT="7c2036704831dc0363a0cafc6bbb85a6d67c1d90"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$SCRIPT_DIR/llama.cpp" ]; then
    echo "Cloning llama.cpp at commit $LLAMACPP_COMMIT..."
    git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$SCRIPT_DIR/llama.cpp"
    cd "$SCRIPT_DIR/llama.cpp"
    git checkout "$LLAMACPP_COMMIT"
    echo "Applying deterministic parity patches..."
    git am "$SCRIPT_DIR/patches/"*.patch
else
    cd "$SCRIPT_DIR/llama.cpp"
fi

echo "Building CPU deterministic verifier..."
cmake -B build \
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
    -DCMAKE_CXX_FLAGS="-mfma"
cmake --build build --target llama-det-diag -j$(nproc)

echo ""
echo "Built: llama.cpp/build/bin/llama-det-diag"
echo ""
echo "Usage:"
echo "  ./verify.sh models/qwen2.5-3b-instruct-q4_k_m.gguf 2024 0.7 \"Once upon a time\""
