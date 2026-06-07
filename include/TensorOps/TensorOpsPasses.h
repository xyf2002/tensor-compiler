//===- TensorOpsPasses.h - TensorOps pass declarations ----------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_OPS_PASSES_H
#define TENSOR_OPS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace ten {

//===----------------------------------------------------------------------===//
// Conversion: TensorOps -> Linalg
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createLowerToLinalgPass();

//===----------------------------------------------------------------------===//
// Operator Fusion: matmul + relu
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createFuseMatmulReluPass();

//===----------------------------------------------------------------------===//
// Pipeline: Linalg -> LLVM
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

//===----------------------------------------------------------------------===//
// Pipeline builder (lowering + fusion)
//===----------------------------------------------------------------------===//

void registerTensorOpsPasses();

} // namespace ten

#endif // TENSOR_OPS_PASSES_H
