#!/bin/bash
# Build the CPU deterministic verifier from source (no Docker needed)
set -euo pipefail

LLAMACPP_COMMIT="d092e2682cc1db9f33b158b4378b448897b3096c"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$SCRIPT_DIR/llama.cpp" ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp.git "$SCRIPT_DIR/llama.cpp"
    cd "$SCRIPT_DIR/llama.cpp"
    echo "Checking out commit $LLAMACPP_COMMIT..."
    git fetch origin "$LLAMACPP_COMMIT" || git fetch --unshallow 2>/dev/null || true
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
