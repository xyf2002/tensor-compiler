#!/bin/bash
# compile_ptx.sh - MLIR TensorOps → PTX cubin compilation pipeline
# Usage: compile_ptx.sh input.mlir output.cubin
#
# Pipeline:
#   tensor-opt (lower TensorOps to NVVM/LLVM)
#     → mlir-opt (lower affine/scf/cf/arith/func/memref to LLVM)
#     → Python (extract gpu.module into standalone module)
#     → mlir-translate (MLIR → LLVM IR)
#     → llc (LLVM IR → PTX)
#     → ptxas (PTX → cubin)

set -euo pipefail

TENSOR_OPT=/home/xyf/tensor-compiler/build/tools/tensor-opt/tensor-opt
MLIR_OPT=/usr/lib/llvm-20/bin/mlir-opt
MLIR_TRANSLATE=/usr/lib/llvm-20/bin/mlir-translate
LLC=/usr/lib/llvm-20/bin/llc
PTXAS=ptxas

INPUT=$1
OUTPUT=$2
TMP_PTX=$(mktemp /tmp/gpu_XXXXXX.ptx)
TMP_MLIR=$(mktemp /tmp/gpu_XXXXXX.mlir)

# Cleanup temp files on exit
trap 'rm -f $TMP_PTX $TMP_MLIR' EXIT

# Step 1: Lower TensorOps to NVVM, then lower to LLVM dialect
$TENSOR_OPT $INPUT -tensor-to-linalg -lower-tensor-ops-to-gpu-nvvm \
  | $MLIR_OPT \
    --lower-affine \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-arith-to-llvm \
    --finalize-memref-to-llvm \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts \
    -o $TMP_MLIR

# Step 2: Extract GPU kernel module (gpu.module → standalone module)
python3 -c "
import re, sys
with open('$TMP_MLIR', 'r') as f:
    lines = f.readlines()
in_gpu = False
depth = 0
out = []
for line in lines:
    if not in_gpu:
        m = re.match(r'(\s*)gpu\.module\s+@(\w+)\s*\{', line)
        if m:
            in_gpu = True
            depth = 1
            out.append('module {\n')
            continue
    if in_gpu:
        depth += line.count('{') - line.count('}')
        if depth <= 0:
            out.append('}\n')
            break
        out.append(line)
sys.stdout.write(''.join(out))
" > $TMP_MLIR.kernel

# Step 3: Translate MLIR (LLVM + NVVM dialect) to LLVM IR
$MLIR_TRANSLATE --mlir-to-llvmir $TMP_MLIR.kernel \
  | $LLC -march=nvptx64 -mcpu=sm_89 -o $TMP_PTX

# Step 4: Assemble PTX to cubin
$PTXAS -arch=sm_89 $TMP_PTX -o $OUTPUT

echo "Generated: $OUTPUT ($(wc -c < $OUTPUT) bytes)"
