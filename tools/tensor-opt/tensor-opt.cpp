//===- tensor-opt.cpp - MLIR Optimizer for TensorOps dialect --------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

#include "TensorOps/TensorOpsDialect.h"
#include "TensorOps/TensorOpsPasses.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerAllDialects(registry);

  // Register our custom TensorOps dialect
  registry.insert<tensor::TensorOpsDialect>();

  // Register our custom passes
  tensor::registerTensorOpsPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TensorOps optimizer driver\n", registry));
}
