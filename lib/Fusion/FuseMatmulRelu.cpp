//===- FuseMatmulRelu.cpp - Matmul + ReLU fusion pattern ------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//
//
// This pass fuses a tensor.matmul followed by tensor.relu into a single
// fused operation by pattern-matching the sequence.
//
//===----------------------------------------------------------------------===//

#include "TensorOps/TensorOpsOps.h"
#include "TensorOps/TensorOpsPasses.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// FuseMatmulReluPattern
//   tensor.matmul %a, %b -> %mm
//   tensor.relu %mm    -> %r
//
//   After fusion:
//   tensor.matmul %a, %b -> %r  (with fused relu attribute)
//===----------------------------------------------------------------------===//

struct FuseMatmulReluPattern : public OpRewritePattern<tensor::ReluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ReluOp reluOp,
                                PatternRewriter &rewriter) const override {
    // Check if the input to relu comes from a matmul
    auto matmulOp = reluOp.getInput().getDefiningOp<tensor::MatmulOp>();
    if (!matmulOp)
      return failure();

    // Check that matmul result is only used by this relu
    if (!matmulOp.getResult().hasOneUse())
      return failure();

    // Create a new matmul with a fused attribute marker
    // The actual fusion will be handled at the linalg level during lowering
    auto fusedOp = rewriter.create<tensor::MatmulOp>(
        reluOp.getLoc(), reluOp.getType(),
        matmulOp.getLhs(), matmulOp.getRhs());

    // Add fused attribute
    fusedOp->setAttr("fused_relu", rewriter.getBoolAttr(true));

    // Replace both the matmul and relu with the fused op
    rewriter.replaceOp(reluOp, fusedOp.getResult());
    rewriter.eraseOp(matmulOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// FuseMatmulReluPass
//===----------------------------------------------------------------------===//

struct FuseMatmulReluPass
    : public PassWrapper<FuseMatmulReluPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseMatmulReluPass)

  StringRef getArgument() const override { return "fuse-matmul-relu"; }
  StringRef getDescription() const override {
    return "Fuse tensor.matmul + tensor.relu into a single operation";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<FuseMatmulReluPattern>(ctx);

    // Use greedy pattern applicator to find and fuse all matmul+relu pairs
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> tensor::createFuseMatmulReluPass() {
  return std::make_unique<FuseMatmulReluPass>();
}
