//===- TensorOpsPasses.cpp - TensorOps pass registration ------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

#include "TensorOps/TensorOpsPasses.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace ten {

void registerTensorOpsPasses() {
  // Register both passes
  mlir::registerPass(createLowerToLinalgPass);
  mlir::registerPass(createFuseMatmulReluPass);
}

} // namespace ten
