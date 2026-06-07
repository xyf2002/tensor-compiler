## [Initial] Learnings from Wave 1

### Naming Conventions
- Dialect name: "ten" (not "tensor" - avoids MLIR built-in tensor dialect conflict)
- C++ namespace: `::ten`
- Pass names: kebab-case (e.g., `tensor-to-linalg`, `fuse-matmul-relu`)
- Ops prefixed with `ten.` (e.g., `ten.matmul`, `ten.relu`, `ten.add`)

### CMake Patterns
- Use `find_package(MLIR REQUIRED CONFIG)` + `find_package(LLVM REQUIRED CONFIG)`
- Link against individual MLIR libs (e.g., `MLIRArithDialect`, `MLIRLinalgDialect`)
- TableGen: `mlir_tablegen(... -gen-dialect-decls/-defs)` and `mlir_tablegen(... -gen-op-decls/-defs)`
- Lib target: `add_public_tablegen_target(TensorOpsIncGen)` + `add_dependencies(TensorOps TensorOpsIncGen)`

### Current Pass Architecture
- `tensor-to-linalg`: Converts ten.matmul → linalg.matmul, ten.relu → linalg.generic, ten.add → linalg.add
- `fuse-matmul-relu`: Fuses ten.matmul + ten.relu at TensorOps dialect level (before lowering)
- `lower-tensor-ops-to-llvm`: Composes passes: one-shot-bufferize → linalg-to-loops → scf-to-cf → memref-to-llvm → func-to-llvm → reconcile-unrealized-casts
- All three passes registered via `ten::registerTensorOpsPasses()`

### Pass Composition Patterns
- For pipelines that compose existing passes, use `PassManager` + `pm.run(mod)` (not `runPipeline`) when the pass operates on `ModuleOp`
- `PassManager` approach: create `PassManager(&getContext())`, add passes, then call `pm.run(mod)`
- Bufferization (`createOneShotBufferizePass`) is required before `createConvertLinalgToLoopsPass` because tensor-to-linalg produces tensor-based linalg ops
- `PassManager` approach requires `#include "mlir/Pass/PassManager.h"`
- Bufferization requires `#include "mlir/Dialect/Bufferization/Transforms/Passes.h"`

### GPU Toolchain
- CUDA toolkit installed via apt: ptxas, nvcc available
- libcuda.so accessible at /usr/lib/x86_64-linux-gnu/libcuda.so
- cuda.h available at /usr/include/cuda.h
- RTX 4080 = sm_89 (Ada Lovelace)

### Test File Creation (Task 10)
- Created 4 FileCheck test files in `tests/`: `conversion.mlir`, `fusion.mlir`, `cpu_lowering.mlir`, `gpu_nvvm.mlir`
- All 5 tests pass: smoke.mlir (existing) + 4 new tests = 5/5 PASS
- Key gotchas encountered:
  - **CHECK-LABEL substring matching**: `CHECK-LABEL: func @test_cpu` matches `func @test_cpu_relu` — use unique function names or `func.func @name` to avoid ambiguity
  - **CHECK ordering matters**: In LLVM lowering, `llvm.call @malloc` appears before `llvm.getelementptr` — checks must be in output order
  - **`llvm.func @malloc` at module level**: Module-level declarations appear before `func.func` — cannot be matched via CHECK after CHECK-LABEL (CHECK-LABEL resets scan to after the matched label)
  - **NVVM `nvvm.kernel` attribute**: On very long function attribute lines — prefer matching `nvvm.read.ptx.sreg.ctaid.x` ops instead
  - Pass flag names verified: `--tensor-to-linalg`, `--fuse-matmul-relu`, `--lower-tensor-ops-to-llvm`, `--lower-tensor-ops-to-gpu-nvvm`
- Running tests: `/home/xyf/.local/bin/lit tests/ -v`
- LIT config at `tests/lit.cfg.py` uses `%tensor-opt` → `tensor-opt` binary

## Task 13 - ROCDL Lowering Pass

- Created `lib/LowerTensorOpsToROCDL.cpp` following the NVVM pass pattern
- **Key difference from NVVM**: Use `createLowerGpuOpsToROCDLOpsPass()` (not `createConvertGpuOpsToROCDLOps()` which doesn't exist). This is auto-generated via `GEN_PASS_DECL_CONVERTGPUOPSTOROCDLOPS` + `#include "mlir/Conversion/Passes.h.inc"` and is available through `mlir/Conversion/Passes.h` which includes `mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h`
- CMake link deps needed: `MLIRGPUToROCDLTransforms` and `MLIRROCDLDialect`
- Pass argument: `lower-tensor-ops-to-gpu-rocdl`, factory: `ten::createLowerToROCDLPass()`
- Pipeline: OneShotBufferize → LinalgGeneralize → LinalgToParallelLoops → GpuMapParallelLoops → ParallelLoopToGpu → GpuKernelOutlining → LowerGpuOpsToROCDLOpsPass → ReconcileUnrealizedCasts
- Verification: `--help | grep rocdl` shows the pass, and running on a tensor matmul test produces `rocdl.*` ops inside a `gpu.module`
