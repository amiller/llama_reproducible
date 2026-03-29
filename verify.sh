#!/bin/bash
# Verify a deterministic seed: given (model, seed, temp, prompt), run CPU
# inference and print the output. Same seed → same tokens on any x86-64 machine.
#
# Usage: ./verify.sh <model.gguf> <seed> <temp> "<prompt>" [n_tokens]
set -euo pipefail

MODEL="${1:?Usage: $0 <model.gguf> <seed> <temp> \"prompt\" [n_tokens]}"
SEED="${2:?}"
TEMP="${3:?}"
PROMPT="${4:?}"
N="${5:-50}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIAG="$SCRIPT_DIR/llama.cpp/build/bin/llama-det-diag"

if [ ! -f "$DIAG" ]; then
    echo "Build first: ./build.sh"
    exit 1
fi

echo "Model:  $(basename "$MODEL")"
echo "Seed:   $SEED"
echo "Temp:   $TEMP"
echo "Prompt: \"$PROMPT\""
echo ""

"$DIAG" -m "$MODEL" -p "$PROMPT" -n "$N" \
    --seed "$SEED" --temp "$TEMP" --deterministic \
    2>&1 >/dev/null | grep "^Token" | \
    sed "s/Token [0-9]*: id=[0-9]* '//" | sed "s/'$//" | tr -d '\n'
echo
