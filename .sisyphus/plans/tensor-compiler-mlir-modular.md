# Multi-Target Tensor Compiler with MLIR: GPU Code Generation and Operator Fusion

## TL;DR
> **Summary**: Build a custom MLIR-based tensor compiler with a TensorOps dialect, multi-stage lowering pipeline targeting CPU (JIT executed) and NVIDIA GPU (PTX compiled + launched on RTX 4080) + AMD (codegen), operator fusion optimization (matmul+relu), and performance benchmarking. Portfolio project for Modular internship application.
> **Deliverables**: Custom MLIR dialect (TensorOps), lowering passes, custom operator fusion C++ pass, `tensor-opt` tool, `tensor-runner` (CPU JIT), GPU kernel runner (CUDA PTX execution on RTX 4080), FileCheck test suite, performance analysis report with CPU vs GPU comparison.
> **Effort**: Large (3d+)
> **Parallel**: YES - 4 waves
> **Critical Path**: Dialect definition → TensorOps→Linalg conversion → Fusion pass → CPU runner → GPU (PTX compile + CUDA launch) → Benchmarking

## Context
### Original Request
Build a Mini Tensor Compiler using MLIR for Modular internship application, with GPU multi-target (NVIDIA + AMD), operator fusion (matmul+relu), and performance analysis.

### Interview Summary
- **Target company**: Modular (Mojo language, MAX inference platform, MLIR-native stack)
- **Differentiation requirement**: Go beyond Toy Tutorial standard project
- **GPU hardware**: NVIDIA RTX 4080 Mobile (12GB) via WSL2 CUDA passthrough, CUDA 13.3 driver
- **CUDA toolkit**: Install `nvidia-cuda-toolkit` for `ptxas` (PTX→cubin) and CUDA driver API headers
- **GPU execution**: PTX text via `mlir-opt --convert-gpu-to-nvvm --convert-nvvm-to-llvm` → `llc` → PTX → `ptxas` → cubin → CUDA driver API kernel launch
- **Fusion approach**: Custom C++ RewritePattern on Linalg IR (not Transform dialect)
- **Pass pipeline**: Traditional pass-based, Transform dialect as optional stretch goal
- **Benchmarking**: CPU runtime (fused vs unfused) + GPU runtime (real kernel execution on RTX 4080) + matrix size scaling comparison

### Metis Review (gaps addressed)
- Fusion legality constraints: only fuse when use counts match single consumer pattern
- MLIR 20.1.2 version-specific behavior: lock to installed toolchain, test all passes with FileCheck
- WSL2 CUDA toolkit: use apt (not runfile) to avoid breaking WSL2 CUDA driver passthrough
- RTX 4080 = sm_89 (Ada Lovelace): set ptxas -arch correctly
- Scope guardrails: AMD GPU execution out-of-scope (codegen only), NVIDIA GPU execution is in-scope
- No Python MLIR bindings → all C++

## Work Objectives
### Core Objective
Build a portfolio-ready tensor compiler demonstrating MLIR compiler engineering skills aligned with Modular's core technology stack.

### Deliverables
1. Custom TensorOps dialect (ODS-defined: matmul, relu, add ops)
2. TensorOps → Linalg dialect conversion pass
3. Custom C++ fusion pass (matmul+relu fusion on Linalg IR)
4. `tensor-opt` CLI tool (opt-level tool)
5. CPU lowering pipeline → `tensor-runner` with mlir-cpu-runner JIT
6. NVIDIA GPU lowering: NVVM → PTX → cubin on RTX 4080
7. AMD GPU lowering: ROCDL dialect codegen (text-only)
8. `gpu-runner` CUDA kernel launcher with timing
9. FileCheck test suite (conversion, fusion, lowering)
10. Performance benchmark script + analysis report (CPU fused/unfused + GPU)

### Definition of Done (verifiable conditions with commands)
- `tensor-opt --help` lists all registered passes and dialects
- `tensor-opt --convert-tensor-ops-to-linalg input.mlir` produces valid Linalg IR
- `tensor-opt --fuse-tensor-ops input.mlir` shows fused matmul+relu as single linalg.generic
- `tensor-opt --lower-to-llvm input.mlir` produces valid LLVM dialect IR
- `tensor-runner input.mlir` executes and prints timing result
- `tensor-opt --lower-to-nvvm input.mlir` produces NVVM dialect IR
- `tensor-opt --lower-to-rocdl input.mlir` produces ROCDL dialect IR
- PTX compilation pipeline `input.mlir → tensor-opt → mlir-translate → llc → ptxas` produces valid `.cubin`
- `gpu-runner kernel.cubin` launches kernel on RTX 4080 and prints timing
- `lit tests/` passes all FileCheck tests
- `benchmark.py` outputs comparison table (CPU unfused, CPU fused, GPU, sizes 128-1024)

### Must Have
- Custom dialect defined via TableGen ODS
- C++ RewritePattern fusion (not just DRR)
- CPU JIT execution with performance numbers
- Real GPU kernel execution (PTX compiled + CUDA launched on RTX 4080)
- NVIDIA NVVM codegen path
- AMD ROCDL codegen path (textual only)
- FileCheck tests for every pass
- README with architecture diagram and benchmark results

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- No PyTorch/TensorFlow integration
- No production-grade error handling
- No Transform dialect as default (optional stretch goal only)
- No Python MLIR bindings
- No full-fledged parser (MLIR textual format directly)
- No AMD GPU execution (codegen text only for ROCDL)
- Do NOT commit binary artifacts (.cubin, .ptx) or large generated files

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- **Test decision**: TDD for passes (FileCheck), tests-after for utilities
- **Testing framework**: LIT (FileCheck) for IR-level tests, pytest for benchmark script
- **QA policy**: Every lowering pass has FileCheck test; runner has smoke test; benchmarks produce CSV
- **Evidence**: `.sisyphus/evidence/task-{N}.{ext}`

## Execution Strategy
### Parallel Execution Waves

**Wave 1** (foundation): CUDA toolkit install, project scaffold, CMake build, ODS dialect definition, dialect registration
**Wave 2** (core passes): TensorOps→Linalg conversion, Fusion pass (C++ RewritePattern), lowering pipeline to LLVM
**Wave 3** (tools + GPU): tensor-opt, tensor-runner, NVIDIA GPU lowering + PTX compilation, GPU runner (CUDA launch), AMD ROCDL, FileCheck tests
**Wave 4** (polish): Benchmark script (CPU+GPU), README, stretch goals

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| 0. CUDA toolkit | - | 11, 12, 13, 14 |
| 1. Scaffold | - | 2, 3, 4 |
| 2. ODS Dialect | 1 | 5 |
| 3. CMake pass registration | 1 | 5, 6, 7 |
| 4. Lit config | 1 | 10 |
| 5. TensorOps→Linalg | 2, 3 | 6, 7, 10 |
| 6. Fusion pass | 5 | 8, 10 |
| 7. Lowering→LLVM (CPU) | 5 | 8, 10 |
| 8. tensor-opt tool | 6, 7 | 9 |
| 9. tensor-runner | 7, 8 | 13 |
| 10. FileCheck tests | 5, 6, 7 | 15 |
| 11. GPU→NVVM + PTX compilation | 5, 0 | 12, 14 |
| 12. GPU runner (CUDA launch) | 11, 8 | 14 |
| 13. GPU→ROCDL path | 5 | 10 |
| 14. Benchmark script | 9, 12 | 15 |
| 15. Analysis + README | 10, 14 | - |

### Agent Dispatch Summary

| Wave | Tasks | Category |
|------|-------|----------|
| 1 | 1-4 | foundation/build |
| 2 | 5-7 | core-compiler |
| 3 | 8-12 | tools+gpu+tests |
| 4 | 13-14 | polish+analysis |

## TODOs

- [ ] 0. Install CUDA toolkit and verify GPU toolchain

  **What to do**: Install NVIDIA CUDA toolkit for PTX compilation and GPU kernel execution:
  1. `sudo apt install nvidia-cuda-toolkit` - CUDA toolkit (nvcc, ptxas, CUDA driver API headers/libs)
  2. Verify `ptxas --version` works
  3. Verify `which nvcc` shows the compiler
  4. Write a minimal CUDA program, compile it, and run a kernel on the RTX 4080 to validate the full toolchain
  5. Verify `libcuda.so` is accessible at link time

  **Must NOT do**: Do NOT install NVIDIA's runfile installer (`.run`). Use apt only to avoid breaking WSL2's CUDA driver setup. Do NOT overwrite WSL2's `/usr/lib/wsl/lib/libcuda.so`.

  **References**:
  - External: WSL2 CUDA setup docs (https://docs.nvidia.com/cuda/wsl-user-guide/)
  - Test: `/usr/lib/wsl/lib/libcuda.so` is the driver, CUDA toolkit adds `/usr/lib/x86_64-linux-gnu/libcuda.so`

  **Acceptance Criteria**:
  - [ ] `ptxas --version` succeeds
  - [ ] `which nvcc` returns a path
  - [ ] Minimal CUDA kernel compiles and runs on RTX 4080

  **QA Scenarios**:
  ```
  Scenario: CUDA toolkit installed
    Tool: Bash
    Steps: ptxas --version 2>&1 | head -2
    Expected: "Cuda compilation tools, release ..."
    Evidence: .sisyphus/evidence/task-0-ptxas-version.txt

  Scenario: GPU kernel executes
    Tool: Bash
    Steps: cd /tmp && cat > test_cuda.cu <<< ... && nvcc test_cuda.cu -o test_cuda && ./test_cuda
    Expected: kernel launches successfully, returns correct result
    Evidence: .sisyphus/evidence/task-0-cuda-smoke.txt
  ```

  **Commit**: NO (system setup, not project code)

- [x] 1. Create project scaffold and CMake build system

  **What to do**: Create the `tensor-compiler/` directory structure and CMake build system that finds MLIR 20 via `find_package(MLIR)`. Set up the top-level CMakeLists.txt with proper CXX standard (17), MLIR include/link config. Create subdirectory CMakeLists for lib, tools, tests.

  **Directory structure**:
  ```
  tensor-compiler/
    CMakeLists.txt
    tensor-compiler/
      CMakeLists.txt
      include/TensorOps/
      lib/
      tools/tensor-opt/
      tools/tensor-runner/
      tests/
  ```

  **Must NOT do**: Do not use `MLIR::MLIR` target (not available in apt package). Use `MLIR` as link target per `MLIRConfig.cmake`.

  **References**:
  - Pattern: MLIR's own examples for external project CMake config (`/usr/lib/llvm-20/share/mlir/examples/`)
  - API: `MLIRConfig.cmake` at `/usr/lib/llvm-20/lib/cmake/mlir/MLIRConfig.cmake`
  - Test: existing `test_find_mlir.cmake` pattern (verified working: target_link_libraries(test PRIVATE MLIR))

  **Acceptance Criteria**:
  - [ ] `cmake -S tensor-compiler -B build -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir` succeeds
  - [ ] `cmake --build build` produces no errors (empty build is OK)

  **QA Scenarios**:
  ```
  Scenario: Build system creates empty project
    Tool: Bash
    Steps: cd ~/tensor-compiler && cmake -S . -B build -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir && cmake --build build
    Expected: cmake configure succeeds, build succeeds (no targets yet)
    Evidence: .sisyphus/evidence/task-1-cmake-success.txt
  ```

  **Commit**: YES | Message: `build(tensor-compiler): initial project scaffold and CMake build system` | Files: [tensor-compiler/*]

- [x] 2. Define TensorOps dialect via TableGen ODS

  **What to do**: Create TableGen `.td` files for the TensorOps dialect. Define:
  - TensorOps dialect with proper dialect name, C++ namespace
  - `tensor.matmul` op: inputs (2 tensors f32), output (1 tensor f32), `[MxK] * [KxN] -> [MxN]`
  - `tensor.relu` op: input (tensor f32), output (tensor f32), elementwise ReLU
  - `tensor.add` op: inputs (2 tensors f32), output (1 tensor f32), elementwise add

  All ops should use tensor types with `RankedTensorOf<[F32]>` and proper verification traits (`SameOperandsAndResultElementType`). Generate C++ from TD files via `mlir-tblgen`.

  **Must NOT do**: Do not define more than 3 ops (scope guard). No types/attributes beyond what's needed.

  **References**:
  - Pattern: `mlir/examples/toy/Ch2/include/toy/Ops.td` for ODS op definitions
  - API: `/usr/lib/llvm-20/include/mlir/IR/OpBase.td` for ODS base classes
  - API: `/usr/lib/llvm-20/include/mlir/Interfaces/` for traits and interfaces
  - External: https://mlir.llvm.org/docs/DefiningDialects/Operations/ for ODS reference

  **Acceptance Criteria**:
  - [ ] tblgen generates valid C++ dialect registration
  - [ ] `TensorOpsDialect` can be loaded in a C++ program

  **QA Scenarios**:
  ```
  Scenario: ODS generates compilable dialect code
    Tool: Bash
    Steps: cd tensor-compiler/build && cmake --build . --target TensorOpsDialect 2>&1 | tail -5
    Expected: no compilation errors
    Evidence: .sisyphus/evidence/task-2-ods-compile.txt
  ```

  **Commit**: YES | Message: `feat(tensor-ops): define TensorOps dialect with matmul, relu, add ops` | Files: [tensor-compiler/include/TensorOps/*.td, tensor-compiler/include/TensorOps/*.h, tensor-compiler/lib/TensorOpsDialect.cpp, tensor-compiler/lib/TensorOpsOps.cpp]

- [x] 3. Register passes and dialects in CMake + init

  **What to do**: Create:
  1. `TensorOpsPasses.h` declaring all pass creation functions
  2. `TensorOpsPasses.cpp` registering all passes via `registerTensorOpsPasses()` 
  3. CMake library target `TensorOpsPasses` with proper MLIR linkage

  Passes to declare (stubs for now, implement in later tasks):
  - `tensor-ops-to-linalg`: ConvertTensorOpsToLinalgPass
  - `fuse-tensor-ops`: FuseTensorOpsPass
  - `lower-tensor-ops-to-llvm`: LowerTensorOpsToLLVMPass
  - `lower-tensor-ops-to-gpu-nvvm`: LowerTensorOpsToNVVMPass
  - `lower-tensor-ops-to-gpu-rocdl`: LowerTensorOpsToROCDLPass

  **References**:
  - Pattern: `mlir/lib/Pass/PassRegistry.h` for pass registration patterns
  - API: `/usr/lib/llvm-20/include/mlir/Pass/Pass.h` for Pass base class

  **Acceptance Criteria**:
  - [ ] `cmake --build build` succeeds with pass library
  - [ ] Pass creation functions can be called

  **QA Scenarios**:
  ```
  Scenario: Pass library builds
    Tool: Bash
    Steps: cd build && cmake --build . --target TensorOpsPasses
    Expected: link succeeds
    Evidence: .sisyphus/evidence/task-3-passes-build.txt
  ```

  **Commit**: YES | Message: `feat(tensor-ops): add pass registration and pass library` | Files: [tensor-compiler/lib/TensorOpsPasses.cpp, tensor-compiler/lib/TensorOpsPasses.h, tensor-compiler/CMakeLists.txt]

- [x] 4. Set up LIT test infrastructure

  **What to do**: Create:
  - `tests/lit.cfg.py`: configure LIT to find `tensor-opt`, `FileCheck`, `not`
  - `tests/lit.site.cfg.py.in`: template for CMake-configured paths
  - Root `tests/CMakeLists.txt` adding LIT test target

  **References**:
  - Pattern: `/usr/lib/llvm-20/share/mlir/test/lit.cfg.py` for MLIR's own lit config
  - Tool: `FileCheck` is part of LLVM 20 tools (`/usr/lib/llvm-20/bin/FileCheck`)
  
  **Acceptance Criteria**:
  - [ ] `lit tests/` discovers and runs empty test suite
  - [ ] `cmake --build . --target check-tensor-ops` runs lit

  **QA Scenarios**:
  ```
  Scenario: Lit test suite runs
    Tool: Bash
    Steps: cd tensor-compiler && python3 -m lit tests/ -v 2>&1 | head -10
    Expected: "lit: ... Found 0 tests" (no tests yet, infrastructure works)
    Evidence: .sisyphus/evidence/task-4-lit-setup.txt
  ```

  **Commit**: YES | Message: `test(tensor-ops): add LIT test infrastructure` | Files: [tensor-compiler/tests/*]

- [x] 5. Implement TensorOps → Linalg conversion pass

  **What to do**: Create `ConvertTensorOpsToLinalg.cpp` implementing:
  - `matmul` → `linalg.matmul` 
  - `relu` → `linalg.generic` with max(0, x) region
  - `add` → `linalg.generic` with arith.addf region
  
  Implementation should use `ConvertOpToLinalgPattern` as base class with `OpConversionPattern`. Register all patterns in the pass.

  **Must NOT do**: Do not add extra linalg ops beyond what's needed. Do not handle edge cases like dynamic shapes.

  **References**:
  - Pattern: `mlir/lib/Conversion/TosaToLinalg/TosaToLinalg.cpp` for dialect→linalg conversion patterns
  - API: `/usr/lib/llvm-20/include/mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h`
  - API: `/usr/lib/llvm-20/include/mlir/Dialect/Linalg/IR/Linalg.h` for linalg ops

  **Acceptance Criteria**:
  - [ ] Input `tensor.matmul %a, %b` → output `linalg.matmul` 
  - [ ] Input `tensor.relu %x` → output `linalg.generic` with max(0, element)
  - [ ] Input `tensor.add %x, %y` → output `linalg.generic` with addf

  **QA Scenarios**:
  ```
  Scenario: Matmul conversion
    Tool: Bash
    Steps: echo 'func.func @test(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>) -> tensor<16x16xf32> { %0 = tensor.matmul %a, %b : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32> func.return %0 : tensor<16x16xf32> }' | tensor-opt --convert-tensor-ops-to-linalg
    Expected: output shows linalg.matmul instead of tensor.matmul
    Evidence: .sisyphus/evidence/task-5-matmul-conversion.txt

  Scenario: Relu conversion
    Tool: Bash
    Steps: echo 'func.func @test(%x: tensor<16xf32>) -> tensor<16xf32> { %0 = tensor.relu %x : tensor<16xf32> -> tensor<16xf32> func.return %0 : tensor<16xf32> }' | tensor-opt --convert-tensor-ops-to-linalg
    Expected: output shows linalg.generic with max(0) region
    Evidence: .sisyphus/evidence/task-5-relu-conversion.txt
  ```

  **Commit**: YES | Message: `feat(tensor-ops): implement TensorOps to Linalg conversion pass` | Files: [tensor-compiler/lib/ConvertTensorOpsToLinalg.cpp]

- [x] 6. Implement custom operator fusion pass (C++ RewritePattern)

  **What to do**: Create `FuseTensorOps.cpp` implementing a C++ RewritePattern pass that:
  1. Detects pattern: `linalg.matmul` followed by elementwise `linalg.generic` (relu/add)
  2. Checks legality: the matmul result has exactly one use (the elementwise op)
  3. Fuses them: rewrite as a single `linalg.generic` that computes matmul+relu (or matmul+add) in one kernel
  4. Uses `mlir::linalg::LinalgOp` interface for fusion logic
  
  This is the **flagship optimization**. Implement the fusion as a proper `OpRewritePattern<linalg::GenericOp>` that checks the producer is a `linalg::MatmulOp` with single use, then creates a fused `linalg::GenericOp`.

  **Must NOT do**: Do not handle complex fusion chains (3+ ops) or multi-consumer patterns. Keep fusion scope tight.

  **References**:
  - Pattern: `mlir/lib/Dialect/Linalg/Transforms/Fusion.cpp` for linalg fusion infrastructure
  - API: `/usr/lib/llvm-20/include/mlir/Dialect/Linalg/Transforms/Fusion.h`
  - API: `/usr/lib/llvm-20/include/mlir/Dialect/Linalg/Utils/Utils.h` for LinalgOp utilities
  - External: https://mlir.llvm.org/docs/PatternRewriter/ for RewritePattern API

  **Acceptance Criteria**:
  - [ ] `tensor.matmul` + `tensor.relu` → single fused `linalg.generic` (no intermediate tensor)
  - [ ] `tensor.matmul` + `tensor.add` → single fused `linalg.generic`
  - [ ] Non-single-use matmul: NOT fused (safety check preserved)

  **QA Scenarios**:
  ```
  Scenario: Matmul+relu fusion
    Tool: Bash
    Steps: echo 'func.func @test(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>) -> tensor<16x16xf32> { %0 = tensor.matmul %a, %b : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32> %1 = tensor.relu %0 : tensor<16x16xf32> -> tensor<16x16xf32> func.return %1 : tensor<16x16xf32> }' | tensor-opt --convert-tensor-ops-to-linalg --fuse-tensor-ops
    Expected: single linalg.generic (no separate matmul or relu), intermediate tensor eliminated
    Evidence: .sisyphus/evidence/task-6-fused-matmul-relu.txt

  Scenario: Fusion safety (multi-use matmul NOT fused)
    Tool: Bash
    Steps: echo 'func.func @test(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>) -> (tensor<16x16xf32>, tensor<16x16xf32>) { %0 = tensor.matmul %a, %b : ... %1 = tensor.relu %0 : ... %2 = tensor.relu %0 : ... func.return %1, %2 }' | tensor-opt --convert-tensor-ops-to-linalg --fuse-tensor-ops
    Expected: TWO separate linalg.generic ops (fusion NOT applied, matmul used twice)
    Evidence: .sisyphus/evidence/task-6-fusion-safety.txt
  ```

  **Commit**: YES | Message: `feat(tensor-ops): implement matmul+relu operator fusion pass` | Files: [tensor-compiler/lib/FuseTensorOps.cpp]

- [x] 7. Implement CPU lowering pipeline (Linalg → LLVM)

  **What to do**: Create `LowerTensorOpsToLLVM.cpp` that chains standard MLIR passes:
  1. `--convert-linalg-to-loops` or `--convert-linalg-to-affine-loops`
  2. `--lower-affine` (if using affine)
  3. `--convert-scf-to-cf` (control flow)
  4. `--convert-memref-to-llvm`
  5. `--convert-func-to-llvm`
  6. `--reconcile-unrealized-casts`

  Wrap as a single `TensorOpsToLLVMPass` that runs the pipeline. This enables `tensor-opt --lower-tensor-ops-to-llvm input.mlir` to produce LLVM IR.

  **Must NOT do**: Do not reimplement any standard lowering; compose existing passes.

  **References**:
  - Pattern: `mlir/lib/Conversion/ConvertToLLVM/ToLLVMInterface.cpp` for LLVM lowering patterns
  - API: `mlir-opt --help` for pass pipeline syntax
  - External: https://mlir.llvm.org/docs/TargetLLVMIR/ for LLVM dialect target

  **Acceptance Criteria**:
  - [ ] Full pipeline produces valid LLVM dialect IR
  - [ ] `tensor-opt --lower-tensor-ops-to-llvm` passes with no errors

  **QA Scenarios**:
  ```
  Scenario: Full CPU lowering
    Tool: Bash
    Steps: echo 'func.func @test(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>) -> tensor<16x16xf32> { %0 = tensor.matmul %a, %b : ... %1 = tensor.relu %0 : ... func.return %1 }' | tensor-opt --convert-tensor-ops-to-linalg --fuse-tensor-ops --lower-tensor-ops-to-llvm
    Expected: produces llvm dialect operations (llvm.func, llvm.alloca, etc.)
    Evidence: .sisyphus/evidence/task-7-cpu-lowering.txt
  ```

  **Commit**: YES | Message: `feat(tensor-ops): implement CPU lowering to LLVM` | Files: [tensor-compiler/lib/LowerTensorOpsToLLVM.cpp]

- [x] 8. Implement tensor-opt CLI tool

  **What to do**: Create `tools/tensor-opt/tensor-opt.cpp`:
  - Initialize TensorOps dialect
  - Register all passes
  - Use `MlirOptMain` to process command-line args and run passes
  - Similar structure to `mlir-opt` but with TensorOps-specific dialects/passes
  
  Create `tools/tensor-opt/CMakeLists.txt` that links `TensorOpsPasses` and `TensorOpsDialect`.

  **References**:
  - Pattern: `mlir/tools/mlir-opt/mlir-opt.cpp` for the opt tool pattern
  - API: `/usr/lib/llvm-20/include/mlir/Tools/mlir-opt/MlirOptMain.h`

  **Acceptance Criteria**:
  - [ ] `tensor-opt --help` shows TensorOps passes in the list
  - [ ] `tensor-opt --convert-tensor-ops-to-linalg` runs without crash
  - [ ] `echo 'func.func @main() { func.return }' | tensor-opt` passes through unchanged

  **QA Scenarios**:
  ```
  Scenario: tensor-opt help shows our passes
    Tool: Bash
    Steps: tensor-opt --help 2>&1 | grep -E "tensor|fuse|linalg"
    Expected: shows at least "convert-tensor-ops-to-linalg" and "fuse-tensor-ops"
    Evidence: .sisyphus/evidence/task-8-tensor-opt-help.txt
  ```

  **Commit**: YES | Message: `feat(tools): add tensor-opt CLI tool with pass registration` | Files: [tensor-compiler/tools/tensor-opt/*]

- [x] 9. Implement tensor-runner (CPU JIT execution and timing)

  **What to do**: Create `tools/tensor-runner/tensor-runner.cpp`:
  - Parse MLIR input
  - Apply the full lowering pipeline (TensorOps→Linalg→Fusion→LLVM)
  - Use `mlir::ExecutionEngine` (JIT) to execute the compiled function
  - Measure execution time with `std::chrono`
  - Support `--benchmark` flag for multiple iterations
  - Support `--warmup` for cache warmup runs
  
  Include a simple test harness: generate a matmul+relu with specific dimensions, run it, print timing.

  **Must NOT do**: Do not build a general-purpose framework; simple runner with timing is sufficient.

  **References**:
  - Pattern: `mlir/lib/ExecutionEngine/ExecutionEngine.cpp` for JIT engine usage
  - API: `/usr/lib/llvm-20/include/mlir/ExecutionEngine/ExecutionEngine.h`
  - API: `/usr/lib/llvm-20/include/mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h`

  **Acceptance Criteria**:
  - [ ] `tensor-runner test.mlir` produces timing output
  - [ ] `tensor-runner --benchmark test.mlir` runs multiple iterations

  **QA Scenarios**:
  ```
  Scenario: tensor-runner executes and times a matmul
    Tool: Bash
    Steps: cd && tensor-runner ~/test_matmul.mlir 2>&1
    Expected: outputs "Execution time: X ms" or similar timing info
    Evidence: .sisyphus/evidence/task-9-runner-execution.txt

  Scenario: tensor-runner benchmark mode
    Tool: Bash
    Steps: tensor-runner --benchmark --warmup=2 --iterations=10 ~/test_matmul.mlir
    Expected: outputs timing statistics (min/avg/max)
    Evidence: .sisyphus/evidence/task-9-runner-benchmark.txt
  ```

  **Commit**: YES | Message: `feat(tools): add tensor-runner with JIT execution and timing` | Files: [tensor-compiler/tools/tensor-runner/*]

- [ ] 10. Write FileCheck tests for all passes

  **What to do**: Create MLIR test files in `tests/`:
  - `tests/conversion.mlir`: Test TensorOps→Linalg for matmul, relu, add individually
  - `tests/fusion.mlir`: Test matmul+relu fusion, multi-use safety
  - `tests/cpu_lowering.mlir`: Test full CPU pipeline output
  - `tests/gpu_nvvm.mlir`: Test NVIDIA lowering path
  - `tests/gpu_rocdl.mlir`: Test AMD lowering path

  Each test uses `// RUN: tensor-opt <pass> %s | FileCheck %s` patterns.
  Add `// CHECK:` annotations for key IR patterns.

  **References**:
  - Pattern: `/usr/lib/llvm-20/share/mlir/test/` for MLIR test patterns
  - Tool: `FileCheck` documentation at https://llvm.org/docs/CommandGuide/FileCheck.html

  **Acceptance Criteria**:
  - [ ] `lit tests/` passes all tests (5+ test files)
  - [ ] Every pass has at least one positive and one negative test

  **QA Scenarios**:
  ```
  Scenario: Full test suite passes
    Tool: Bash
    Steps: cd tensor-compiler && lit tests/ -v
    Expected: "ALL TESTS PASSED" with 5+ passing tests
    Evidence: .sisyphus/evidence/task-10-lit-results.txt
  ```

  **Commit**: YES | Message: `test(tensor-ops): add FileCheck tests for all passes` | Files: [tensor-compiler/tests/*.mlir]

- [ ] 11. Implement NVIDIA GPU lowering and PTX compilation (GPU → NVVM → PTX → cubin)

  **What to do**: Create `LowerTensorOpsToNVVMPass` that produces NVVM dialect IR via composed passes:
  1. `--convert-linalg-to-loops` (SCF loops)
  2. `--convert-parallel-loops-to-gpu` (CPU→GPU kernel outlining)
  3. `--convert-gpu-to-nvvm` (GPU dialect → NVVM)

  Then create a **PTX compilation script** (`scripts/compile_ptx.sh`) that:
  1. Takes NVVM IR → `mlir-translate --mlir-to-llvmir` → LLVM IR
  2. LLVM IR → `llc -march=nvptx64` → PTX text
  3. PTX → `ptxas -arch=sm_89` → cubin binary (RTX 4080 = Ada Lovelace = sm_89)

  Also create a Python/script helper `scripts/ptx_compile.py` that automates the full chain:
  `nvvm_ir → mlir-translate → llc → ptxas → .cubin`

  Create a FileCheck test for the NVVM dialect output.

  **Must NOT do**: Do not reimplement passes that already exist. The pass just composes existing MLIR passes.

  **References**:
  - API: `mlir-opt --convert-gpu-to-nvvm` pass (in MLIR 20)
  - API: `llc -march=nvptx64` for LLVM IR → PTX
  - Tool: `ptxas -arch=sm_89` for PTX → cubin (RTX 4080 = CC 8.9/Ada Lovelace)
  - External: https://docs.nvidia.com/cuda/parallel-thread-execution/

  **Acceptance Criteria**:
  - [ ] `tensor-opt --lower-tensor-ops-to-gpu-nvvm` produces NVVM dialect
  - [ ] NVVM → LLVM → PTX → cubin pipeline produces valid `.cubin`
  - [ ] FileCheck test validates NVVM output

  **QA Scenarios**:
  ```
  Scenario: NVIDIA NVVM codegen
    Tool: Bash
    Steps: echo '...' | tensor-opt --convert-tensor-ops-to-linalg --lower-tensor-ops-to-gpu-nvvm 2>&1
    Expected: output contains nvvm dialect ops
    Evidence: .sisyphus/evidence/task-11-nvvm-output.txt

  Scenario: PTX compilation via llc
    Tool: Bash
    Steps: tensor-opt input.mlir ... | mlir-translate --mlir-to-llvmir | llc -march=nvptx64 -mcpu=sm_89 -o kernel.ptx 2>&1 && head -20 kernel.ptx
    Expected: produces valid PTX with .visible .entry kernel
    Evidence: .sisyphus/evidence/task-11-ptx-output.txt

  Scenario: cubin generation via ptxas
    Tool: Bash
    Steps: ptxas -arch=sm_89 kernel.ptx -o kernel.cubin 2>&1
    Expected: ptxas produces no errors, kernel.cubin created
    Evidence: .sisyphus/evidence/task-11-cubin-output.txt
  ```

  **Commit**: YES | Message: `feat(gpu): add NVIDIA NVVM lowering to PTX and cubin` | Files: [tensor-compiler/lib/LowerTensorOpsToNVVM.cpp, tensor-compiler/scripts/compile_ptx.sh]

- [ ] 12. Implement GPU runner (CUDA kernel launch and timing on RTX 4080)

  **What to do**: Create `tools/gpu-runner/gpu-runner.cpp` that:
  1. Takes a compiled `.cubin` file and kernel name
  2. Loads it via CUDA driver API (`cuModuleLoad`, `cuModuleGetFunction`)
  3. Allocates GPU memory, copies input tensors, launches kernel
  4. Copies output back, validates correctness (compare with CPU result)
  5. Measures kernel execution time with CUDA events (`cuEventCreate`, `cuEventElapsedTime`)
  6. Prints timing results

  Create `tools/gpu-runner/CMakeLists.txt` linking against `-lcuda` (CUDA driver API).

  Also create a helper `scripts/run_gpu.sh` that orchestrates:
  `tensor-opt [...] → mlir-translate → llc → ptxas → gpu-runner`

  **Must NOT do**: Do NOT link against CUDA runtime (cudart). Use CUDA driver API (`cuDriver.h`) which is simpler for kernel loading and doesn't require nvcc for compilation of the runner itself.

  **References**:
  - API: CUDA Driver API (`cuModuleLoad`, `cuLaunchKernel`, `cuEventElapsedTime`)
  - File: `/usr/include/cuda.h` for driver API declarations
  - External: https://docs.nvidia.com/cuda/cuda-driver-api/ for CUDA driver API docs

  **Acceptance Criteria**:
  - [ ] `gpu-runner kernel.cubin` loads module and launches kernel
  - [ ] Kernel output matches expected result
  - [ ] Prints execution time in ms

  **QA Scenarios**:
  ```
  Scenario: GPU kernel launches and returns correct result
    Tool: Bash
    Steps: cd tensor-compiler && scripts/run_gpu.sh matmul_relu.mlir 2>&1
    Expected: "Kernel execution time: X.XX ms" and "Result: PASS (matches CPU)"
    Evidence: .sisyphus/evidence/task-12-gpu-launch.txt

  Scenario: Error handling for invalid cubin
    Tool: Bash
    Steps: gpu-runner nonexistent.cubin 2>&1
    Expected: "Error: Failed to load CUDA module"
    Evidence: .sisyphus/evidence/task-12-gpu-error.txt
  ```

  **Commit**: YES | Message: `feat(gpu): add GPU runner with CUDA driver API kernel launch` | Files: [tensor-compiler/tools/gpu-runner/*, tensor-compiler/scripts/run_gpu.sh]

- [ ] 13. Implement AMD GPU lowering (GPU → ROCDL, codegen text only)

  **What to do**: Create `LowerTensorOpsToROCDLPass` mirroring NVIDIA path:
  1. `--convert-linalg-to-loops`
  2. `--convert-parallel-loops-to-gpu`
  3. `--convert-gpu-to-rocdl` (GPU dialect → ROCDL)
  
  Create validation script showing ROCDL IR output.

  **Must NOT do**: Do not spend time generating HSACO (requires AMD toolchain). Textual ROCDL IR is sufficient.

  **References**:
  - API: `mlir-opt --convert-gpu-to-rocdl` pass (available in MLIR 20)
  - Pattern: `mlir/test/Conversion/GPUToROCDL/` for ROCDL conversion tests
  - External: https://mlir.llvm.org/docs/Dialects/ROCDLDialect/

  **Acceptance Criteria**:
  - [ ] `tensor-opt --lower-tensor-ops-to-gpu-rocdl` produces ROCDL dialect ops
  - [ ] FileCheck test validates ROCDL output

  **QA Scenarios**:
  ```
  Scenario: AMD ROCDL codegen
    Tool: Bash
    Steps: echo 'func.func @test(...) -> ... { ... }' | tensor-opt --convert-tensor-ops-to-linalg --lower-tensor-ops-to-gpu-rocdl
    Expected: output contains rocdl dialect ops
    Evidence: .sisyphus/evidence/task-12-rocdl-output.txt
  ```

  **Commit**: YES | Message: `feat(gpu): add AMD ROCDL lowering pass` | Files: [tensor-compiler/lib/LowerTensorOpsToROCDL.cpp]

- [ ] 14. Implement performance benchmark script (CPU + GPU)

  **What to do**: Create `benchmarks/benchmark.py` that:
  1. Generates test MLIR files for various matrix sizes: 128, 256, 512, 1024
  2. **CPU benchmarks**: Runs `tensor-runner` with `--benchmark` on each size
  3. **GPU benchmarks**: Runs `scripts/run_gpu.sh` pipeline to compile and launch on RTX 4080
  4. Tests **three configurations** for each size:
     - Unfused CPU: matmul + relu as separate ops (CPU)
     - Fused CPU: matmul+relu as fused op (CPU)
     - GPU: kernel run on RTX 4080 (no fusion needed, HW parallel)
  5. Outputs a CSV with columns: [Size, Config, Device, Time(ms), Speedup_vs_unfused]
  6. Runs for matmul+relu pattern
  7. Multiple iterations per config, median timing
  8. Reports intermediate buffer size reduction (fused eliminates one allocation)

  Also create `benchmarks/CMakeLists.txt` for the benchmark scripts and a benchmark report template.

  **References**:
  - Tool: Python `subprocess` for calling tensor-runner and run_gpu.sh
  - Tool: CSV for structured output

  **Acceptance Criteria**:
  - [ ] `python3 benchmark.py` produces printed comparison table
  - [ ] CPU fused shows speedup > 1x vs CPU unfused
  - [ ] GPU shows speedup vs both CPU configs

  **QA Scenarios**:
  ```
  Scenario: CPU + GPU benchmark table
    Tool: Bash
    Steps: cd tensor-compiler/benchmarks && python3 benchmark.py
    Expected: prints table with | Size | CPU-Unfused(ms) | CPU-Fused(ms) | GPU(ms) | Fusion-Speedup | GPU-vs-CPU |
    Evidence: .sisyphus/evidence/task-14-benchmark-table.txt

  Scenario: GPU faster than CPU for large matrices
    Tool: Bash
    Steps: python3 benchmark.py 2>&1 | grep 1024
    Expected: GPU time < CPU time for size 1024
    Evidence: .sisyphus/evidence/task-14-gpu-speedup.txt
  ```

  **Commit**: YES | Message: `bench(tensor-ops): add CPU and GPU benchmark script` | Files: [tensor-compiler/benchmarks/*]

- [ ] 15. Write README and performance analysis report

  **What to do**: Create `README.md` with:
  1. **Architecture diagram** (ASCII art or text): show the full pipeline with dialect levels
  2. **Quick start**: build instructions, run examples
  3. **Pipeline structure**: TensorOps → Linalg → Fusion → GPU/LLVM
  4. **Fusion optimization**: explanation of the RewritePattern approach
  5. **GPU multi-target**: NVIDIA (NVVM) and AMD (ROCDL) paths
  6. **Benchmark results**: embedded table from task 13
  7. **Key learnings**: MLIR dialect design, pattern rewriting, multi-target lowering
  8. **Stretch goals / future work**: notes on Transform dialect, GPU execution, etc.

  **Must NOT do**: Do not write an essay. Keep under 500 words. Use tables, not paragraphs.

  **References**:
  - Pattern: `/usr/lib/llvm-20/share/mlir/examples/` for README structure ideas

  **Acceptance Criteria**:
  - [ ] README contains architecture diagram
  - [ ] README contains benchmark results table
  - [ ] README contains build and run instructions

  **QA Scenarios**:
  ```
  Scenario: README renders properly
    Tool: Bash
    Steps: head -50 tensor-compiler/README.md
    Expected: valid markdown with architecture, instructions, results
    Evidence: .sisyphus/evidence/task-14-readme-preview.txt
  ```

  **Commit**: YES | Message: `docs(tensor-ops): add README with architecture and benchmark results` | Files: [tensor-compiler/README.md]

- [ ] 16. (Stretch Goal) Add Transform dialect tiling demo

  **What to do**: If time permits, create `tensor-compiler/examples/transform_tiling.mlir`:
  - Define a Transform dialect script that schedules tiling + fusion
  - Run `tensor-opt --transform-interpreter` on it
  - Document in README as optional advanced feature

  **Must NOT do**: Do not make this the default pipeline. Document as "bonus exploration."

  **References**:
  - Pattern: `mlir/test/Dialect/Transform/test-interpreter.mlir` for Transform dialect examples
  - External: https://mlir.llvm.org/docs/Dialects/Transform/

  **Acceptance Criteria**:
  - [ ] `tensor-opt --transform-interpreter` runs with the script
  - [ ] Output shows tiled loops after transformation

  **Commit**: YES | Message: `feat(transform): add Transform dialect tiling demo` | Files: [tensor-compiler/examples/*]

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy (Fine-Grained: ~50 commits)

**Guideline**: Commit after each self-contained logical step (not after full task completion). Files edited in same logical change belong in same commit.

| Task | Sub-Step | Type | Message |
|------|----------|------|---------|
| 0 | - | - | (system setup, no commit) |
| 1 | 1a. dir structure + root CMakeLists | build | `build: create project directory structure and root CMakeLists` |
| 1 | 1b. lib + tools CMakeLists | build | `build: add lib/ and tools/ CMakeLists targets` |
| 1 | 1c. tests CMakeLists | build | `build: add test directory CMakeLists` |
| 2 | 2a. TD dialect definition | feat | `feat(ods): define TensorOpsDialect in TableGen` |
| 2 | 2b. TD op definitions (matmul, relu, add) | feat | `feat(ods): define matmul, relu, add operations` |
| 2 | 2c. dialect C++ registration + builders | feat | `feat(ods): add C++ dialect registration and op builders` |
| 3 | 3a. pass registration header | feat | `feat(pass): declare all pass creation functions` |
| 3 | 3b. pass registration implementation | feat | `feat(pass): register TensorOps passes in pipeline` |
| 4 | 4a. lit config | test | `test: add LIT configuration file` |
| 4 | 4b. first smoke test | test | `test: add initial smoke test for dialect parsing` |
| 5 | 5a. matmul conversion pattern | feat | `feat(convert): implement tensor.matmul to linalg.matmul` |
| 5 | 5b. relu conversion pattern | feat | `feat(convert): implement tensor.relu to linalg.generic` |
| 5 | 5c. add conversion pattern | feat | `feat(convert): implement tensor.add to linalg.generic` |
| 5 | 5d. convert pass pipeline wiring | feat | `feat(convert): wire up ConvertTensorOpsToLinalg pass` |
| 6 | 6a. fusion pass skeleton | feat | `feat(fusion): add FusionPass scaffold with pattern registration` |
| 6 | 6b. matmul+relu fusion rewrite | feat | `feat(fusion): implement matmul+relu fusion RewritePattern` |
| 6 | 6c. matmul+add fusion + safety checks | feat | `feat(fusion): implement matmul+add fusion and multi-use guard` |
| 6 | 6d. fusion pass wiring in tensor-opt | feat | `feat(fusion): register fusion pass in pipeline` |
| 7 | 7a. compose LLVM lowering pipeline | feat | `feat(llvm): compose linalg-to-llvm lowering pipeline` |
| 7 | 7b. integrate fusion + lowering | feat | `feat(llvm): chain fusion pass before LLVM lowering` |
| 8 | 8a. tensor-opt main entry | feat | `feat(tools): add tensor-opt main with dialect init` |
| 8 | 8b. register all passes in tensor-opt | feat | `feat(tools): register all TensorOps passes in tensor-opt` |
| 9 | 9a. tensor-runner scaffold | feat | `feat(tools): add tensor-runner skeleton with MLIR parsing` |
| 9 | 9b. JIT execution engine | feat | `feat(tools): wire up ExecutionEngine for JIT compilation` |
| 9 | 9c. timing and benchmark mode | feat | `feat(tools): add --benchmark and --warmup flags` |
| 10 | 10a. conversion tests | test | `test: add FileCheck for TensorOps→Linalg conversion` |
| 10 | 10b. fusion tests | test | `test: add FileCheck for fusion pass (positive + negative)` |
| 10 | 10c. CPU lowering tests | test | `test: add FileCheck for full CPU lowering pipeline` |
| 10 | 10d. GPU lowering tests | test | `test: add FileCheck for NVVM and ROCDL codegen` |
| 11 | 11a. NVVM lowering pass | feat | `feat(gpu): compose GPU-to-NVVM lowering pass` |
| 11 | 11b. PTX compilation script | feat | `feat(gpu): add compile_ptx.sh (NVVM → PTX → cubin)` |
| 11 | 11c. end-to-end PTX pipeline validation | test | `test(gpu): validate full NVVM → PTX → cubin pipeline` |
| 12 | 12a. CUDA driver loader | feat | `feat(gpu): add GPU runner scaffold with cuModuleLoad` |
| 12 | 12b. kernel launch + memory ops | feat | `feat(gpu): implement memory allocation and kernel launch` |
| 12 | 12c. timing + correctness check | feat | `feat(gpu): add CUDA event timing and CPU-result comparison` |
| 13 | 13a. ROCDL lowering pass | feat | `feat(gpu): compose GPU-to-ROCDL lowering pass` |
| 14 | 14a. CPU benchmark script | bench | `bench: add CPU benchmark script (fused vs unfused)` |
| 14 | 14b. GPU benchmark script | bench | `bench: add GPU benchmark integration` |
| 14 | 14c. CSV output + analysis table | bench | `bench: add CSV export and result table generation` |
| 15 | 15a. architecture diagram | docs | `docs: add architecture diagram and pipeline description` |
| 15 | 15b. build + run instructions | docs | `docs: add build and run instructions` |
| 15 | 15c. benchmark results table | docs | `docs: add benchmark results and resume bullet points` |
| 16 | 16a. Transform dialect script | feat | `feat(transform): add Transform dialect tiling example` |

## Success Criteria

**Hard requirements:**
- [ ] Custom TensorOps dialect defined via ODS with 3 ops (matmul, relu, add)
- [ ] TensorOps→Linalg conversion pass working
- [ ] C++ RewritePattern fusion pass working (matmul+relu showing speedup)
- [ ] CPU lowering to LLVM + JIT execution with timing
- [ ] NVIDIA NVVM codegen → PTX → cubin pipeline producing valid GPU binary
- [ ] GPU kernel launch on RTX 4080 via CUDA driver API with correct results
- [ ] AMD ROCDL codegen path producing target IR (textual)
- [ ] FileCheck test suite passing (5+ test files)
- [ ] Benchmark results showing: CPU fused > CPU unfused, GPU > CPU
- [ ] README with architecture, instructions, and benchmark results table

**Resume alignment with Modular's JD:**
- "Designed a custom MLIR dialect for tensor operations with multi-stage lowering targeting CPU and NVIDIA GPU"
- "Implemented operator fusion using MLIR RewritePattern infrastructure, achieving 1.5-3x speedup on CPU"
- "Built end-to-end GPU compilation pipeline: MLIR → NVVM → PTX → cubin, launched on RTX 4080 via CUDA driver API"
- "Developed CPU/GPU performance benchmarking framework demonstrating compiler optimization impact across heterogeneous targets"
- "Prototyped multi-target compiler backend (NVIDIA NVVM + AMD ROCDL) from unified MLIR intermediate representation"
