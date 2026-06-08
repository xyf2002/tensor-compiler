# TensorOps - A Multi-Target MLIR Tensor Compiler with Operator Fusion

## Architecture

```
TensorOps Dialect (ten.matmul, ten.relu, ten.add)
          |
          v
TensorOps -> Linalg (--tensor-to-linalg)
          |
     +----+----+
     v         v
   Fusion    Lowering
   (matmul   (bufferize -> linalg->loops -> scf->cf
    +relu)    -> memref->llvm -> func->llvm)
     |         |
     v         v
   Fused     LLVM Dialect -> JIT (lli)
   TensorOps
     |
     v
   GPU Lowering
     |
   +--+--+
   v     v
  NVVM  ROCDL
   |     |
   v     v
  PTX   (codegen
   |    text only)
   v
  cubin -> CUDA Driver API
         (RTX 4080)
```

## Build

```bash
cmake -S . -B build -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
ninja -C build tensor-opt tensor-runner gpu-runner
```

## Usage

| Command | Description |
|---------|-------------|
| `tensor-opt --tensor-to-linalg input.mlir` | Convert TensorOps to Linalg |
| `tensor-opt --fuse-matmul-relu --tensor-to-linalg input.mlir` | Fuse and lower |
| `tensor-runner input.mlir` | JIT execute on CPU |
| `tensor-runner --benchmark --iterations=10 input.mlir` | CPU benchmark |
| `tensor-opt --lower-tensor-ops-to-gpu-nvvm input.mlir` | NVIDIA GPU lowering |
| `tensor-opt --lower-tensor-ops-to-gpu-rocdl input.mlir` | AMD GPU lowering |
| `scripts/compile_ptx.sh input.mlir` | Full PTX -> cubin pipeline |
| `gpu-runner kernel.cubin matmul_kernel 1024` | Launch GPU kernel |
| `scripts/run_gpu.sh input.mlir 1024` | Full GPU pipeline |
| `python3 benchmarks/benchmark.py --sizes 64 128 256 512` | Run benchmarks |

## Passes

| Pass | Description |
|------|-------------|
| `--tensor-to-linalg` | Convert TensorOps to Linalg dialect |
| `--fuse-matmul-relu` | Fuse matmul+relu at TensorOps level |
| `--lower-tensor-ops-to-llvm` | Full CPU lowering pipeline to LLVM |
| `--lower-tensor-ops-to-gpu-nvvm` | NVIDIA GPU lowering to NVVM dialect |
| `--lower-tensor-ops-to-gpu-rocdl` | AMD GPU lowering to ROCDL dialect |

## Benchmark Results (preliminary)

| Size | CPU Unfused(ms) | CPU Fused(ms) | GPU(ms) | Fusion Speedup | GPU vs CPU |
|------|-----------------|---------------|---------|----------------|------------|
| 64   | 24.26           | 24.98         | --      | 0.97x          | --         |
| 128  | 48.63           | 48.52         | --      | 1.00x          | --         |
| 256  | --              | --            | --      | --             | --         |

"--" = pending GPU benchmark (requires CUDA GPU)

## Project Structure

```
include/TensorOps/    ODS dialect definitions, pass headers
lib/                  Pass implementations
tools/                tensor-opt, tensor-runner, gpu-runner
scripts/              PTX compilation, GPU pipeline
benchmarks/           Performance benchmark script
tests/                FileCheck LIT tests
```

## Key Features

- Custom MLIR dialect (TensorOps) with matmul, relu, add operations
- Multi-target code generation: CPU (LLVM JIT), NVIDIA GPU (NVVM->PTX->cubin), AMD GPU (ROCDL)
- Operator fusion at TensorOps dialect level (matmul+relu)
- LIT test suite with FileCheck (6 tests)
- CPU JIT execution via lli subprocess pipeline
- GPU kernel launch via CUDA Driver API
- Performance benchmark framework

## Motivation

Portfolio project demonstrating MLIR dialect design, multi-target lowering, operator fusion, and GPU code generation.
