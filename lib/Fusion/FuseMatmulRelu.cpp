//===- FuseMatmulRelu.cpp - Matmul + ReLU fusion pattern ------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

#include "TensorOps/TensorOpsOps.h"
#include "TensorOps/TensorOpsPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct FuseMatmulReluPattern : public OpRewritePattern<ten::ReluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ten::ReluOp reluOp,
                                PatternRewriter &rewriter) const override {
    auto matmulOp = reluOp.getInput().getDefiningOp<ten::MatmulOp>();
    if (!matmulOp)
      return failure();

    if (!matmulOp.getResult().hasOneUse())
      return failure();

    auto fusedOp = rewriter.create<ten::MatmulOp>(
        reluOp.getLoc(), reluOp.getType(),
        matmulOp.getLhs(), matmulOp.getRhs());
    fusedOp->setAttr("fused_relu", rewriter.getBoolAttr(true));

    rewriter.replaceOp(reluOp, fusedOp.getResult());
    rewriter.eraseOp(matmulOp);

    return success();
  }
};

struct FuseMatmulReluPass
    : public PassWrapper<FuseMatmulReluPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseMatmulReluPass)

  StringRef getArgument() const override { return "fuse-matmul-relu"; }
  StringRef getDescription() const override {
    return "Fuse ten.matmul + ten.relu into a single operation";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    mod.walk([&](func::FuncOp func) {
      RewritePatternSet patterns(ctx);
      patterns.add<FuseMatmulReluPattern>(ctx);
      if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
        // Best-effort — patterns may not match all functions
        return;
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> ten::createFuseMatmulReluPass() {
  return std::make_unique<FuseMatmulReluPass>();
}
