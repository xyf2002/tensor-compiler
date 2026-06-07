# Draft: Tensor Compiler MLIR Project for Modular Internship

## Requirements (confirmed)
- Build a Mini Tensor Compiler using MLIR targeting Modular internship application
- GPU multi-target support (RTX 4080 available via WSL2)
- Operator fusion (matmul+relu)
- Performance analysis and benchmarking (CPU + GPU real execution)
- Differentiated from standard Toy Tutorial
- Project lives in ~/tensor-compiler/

## Technical Decisions
- **MLIR version**: 20.1.2 (already installed via apt)
- **GPU hardware**: NVIDIA RTX 4080 Mobile (12GB) via WSL2
- **CUDA toolkit**: Available via apt (nvidia-cuda-toolkit), need to install
- **GPU execution**: Install CUDA toolkit → compile PTX with ptxas → launch kernel via CUDA driver API
- **Custom dialect**: TensorOps dialect with matmul, relu, add ops (TableGen ODS)
- **Lowering**: Custom TensorOps → Linalg → SCF/Affine → GPU/NVVM/ROCDL
- **Fusion**: Custom C++ RewritePattern for matmul+relu fusion on Linalg IR
- **Performance**: CPU (mlir-cpu-runner JIT timing) + GPU (CUDA kernel execution time), fused vs unfused
- **Testing**: FileCheck for IR transformation verification
- **Transform dialect**: Optional stretch goal

## Research Findings (updated with GPU info)
- WSL2 with CUDA passthrough: RTX 4080, CUDA 13.3 driver, no CUDA toolkit (need to install nvidia-cuda-toolkit)
- MLIR 20 apt package has GPU dialect passes but no CUDA runtime linkage in ExecutionEngine
- GPU execution strategy: emit PTX via mlir-opt pipeline → compile with ptxas → launch with CUDA driver API
- All WSL2 GPU features confirmed available

## Scope Boundaries
- INCLUDE: Custom TensorOps dialect, lowering to Linalg, operator fusion, GPU multi-target lowering, CPU JIT execution, GPU kernel execution (NVIDIA), performance benchmarking, CMake build, FileCheck tests
- EXCLUDE: PyTorch/ONNX integration, AMD GPU execution (codegen only), WebGPU, production optimization
