#!/usr/bin/env bash
#
# run_gpu.sh - Full GPU pipeline: MLIR -> NVVM -> LLVM -> PTX -> cubin -> launch
#
# Usage: run_gpu.sh <input.mlir> [matrix_size]
#
# Pipeline:
#   tensor-opt (lower TensorOps to NVVM)
#     -> mlir-translate (MLIR -> LLVM IR)
#     -> llc (LLVM IR -> PTX)
#     -> ptxas (PTX -> cubin)
#     -> gpu-runner (load cubin and launch kernel)
#
set -euo pipefail

TENSOR_OPT="${TENSOR_OPT:-tensor-opt}"
GPU_RUNNER="${GPU_RUNNER:-gpu-runner}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-mlir-translate}"
LLC="${LLC:-llc}"
PTXAS="${PTXAS:-ptxas}"

INPUT="${1:?Usage: $0 <input.mlir> [matrix_size]}"
MATRIX_SIZE="${2:-64}"
WORKDIR=$(mktemp -d)
trap "rm -rf $WORKDIR" EXIT

BASENAME=$(basename "$INPUT" .mlir)

echo "=== Step 1: Lower to NVVM ==="
$TENSOR_OPT --tensor-to-linalg --lower-tensor-ops-to-gpu-nvvm "$INPUT" -o "$WORKDIR/$BASENAME.nvvm" 2>/dev/null

echo "=== Step 2: NVVM -> LLVM IR ==="
# Extract gpu module and translate to LLVM IR
python3 -c "
import sys, re
with open('$WORKDIR/$BASENAME.nvvm') as f:
    content = f.read()
# Find the GPU module inside the output
parts = content.split('gpu.module @')
if len(parts) < 2:
    print('Error: No gpu.module found', file=sys.stderr)
    sys.exit(1)
# Reconstruct: gpu.module @<name> { ... }
gpu_part = parts[1]
# Find closing brace
depth = 0
end = 0
for i, c in enumerate(gpu_part):
    if c == '{':
        depth += 1
    elif c == '}':
        depth -= 1
        if depth == 0:
            end = i + 1
            break
gpu_module = 'module attributes {gpu.container_module} {\n  gpu.module @kernel {\n' + gpu_part[:end] + '\n  }\n}'
with open('$WORKDIR/$BASENAME.gpu.mlir', 'w') as f:
    f.write(gpu_module)
print('GPU module extracted')
" 2>&1

$MLIR_TRANSLATE --mlir-to-llvmir "$WORKDIR/$BASENAME.gpu.mlir" -o "$WORKDIR/$BASENAME.ll" 2>/dev/null

echo "=== Step 3: LLVM IR -> PTX ==="
$LLC -march=nvptx64 -mcpu=sm_89 "$WORKDIR/$BASENAME.ll" -o "$WORKDIR/$BASENAME.ptx" 2>/dev/null

echo "=== Step 4: PTX -> cubin ==="
$PTXAS -arch=sm_89 "$WORKDIR/$BASENAME.ptx" -o "$WORKDIR/$BASENAME.cubin" 2>/dev/null

echo "=== Step 5: Launch kernel ==="
$GPU_RUNNER "$WORKDIR/$BASENAME.cubin" "matmul_kernel" "$MATRIX_SIZE"
