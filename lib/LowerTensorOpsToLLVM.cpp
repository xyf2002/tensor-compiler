//===- LowerTensorOpsToLLVM.cpp - Linalg -> LLVM lowering pipeline --------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

#include "TensorOps/TensorOpsPasses.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace {

struct LowerTensorOpsToLLVMPass
    : public PassWrapper<LowerTensorOpsToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTensorOpsToLLVMPass)

  StringRef getArgument() const override { return "lower-tensor-ops-to-llvm"; }
  StringRef getDescription() const override {
    return "Lower TensorOps (via Linalg) to LLVM dialect";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Build pipeline: Bufferize -> Linalg->Loops -> SCF->CF -> MemRef->LLVM -> Func->LLVM
    PassManager pm(&getContext());
    pm.addPass(bufferization::createOneShotBufferizePass());
    pm.addPass(createConvertLinalgToLoopsPass());
    pm.addPass(createConvertSCFToCFPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createReconcileUnrealizedCastsPass());

    if (failed(pm.run(mod)))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> ten::createLowerToLLVMPass() {
  return std::make_unique<LowerTensorOpsToLLVMPass>();
}
