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
