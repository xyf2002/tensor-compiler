//===- LowerTensorOpsToROCDL.cpp - Linalg -> ROCDL lowering pipeline -------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

#include "TensorOps/TensorOpsPasses.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace {

struct LowerTensorOpsToROCDLPass
    : public PassWrapper<LowerTensorOpsToROCDLPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTensorOpsToROCDLPass)

  StringRef getArgument() const override {
    return "lower-tensor-ops-to-gpu-rocdl";
  }
  StringRef getDescription() const override {
    return "Lower TensorOps (via Linalg) to ROCDL dialect for AMD GPUs";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    PassManager pm(&getContext());

    // Phase 1: Bufferize with function boundaries
    bufferization::OneShotBufferizationOptions bufOpts;
    bufOpts.bufferizeFunctionBoundaries = true;
    pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));

    // Phase 2: Linalg → parallel loops → GPU launch
    pm.addPass(createLinalgGeneralizeNamedOpsPass());
    pm.addPass(createConvertLinalgToParallelLoopsPass());
    pm.nest<func::FuncOp>().addPass(createGpuMapParallelLoopsPass());
    pm.addPass(createParallelLoopToGpuPass());

    // Phase 3: Kernel outlining → ROCDL
    pm.addPass(createGpuKernelOutliningPass());
    pm.nest<gpu::GPUModuleOp>().addPass(createLowerGpuOpsToROCDLOpsPass());
    pm.nest<gpu::GPUModuleOp>().addPass(createReconcileUnrealizedCastsPass());

    if (failed(pm.run(mod)))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> ten::createLowerToROCDLPass() {
  return std::make_unique<LowerTensorOpsToROCDLPass>();
}
